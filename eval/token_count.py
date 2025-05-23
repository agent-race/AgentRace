from weave.trace.weave_client import WeaveClient, map_to_refs, sum_dict_leaves
from weave.trace.op import _IteratorWrapper

from weave.trace.context import weave_client_context as weave_client_context
from weave.trace.context.tests_context import get_raise_on_captured_errors
from weave.trace import trace_sentry, urls
import datetime
from typing import Any
import pydantic
from weave.trace.context import call_context
from weave.trace.context import weave_client_context as weave_client_context
from weave.trace.exception import exception_to_json_str
from weave.trace.op import (
    Op,
    is_placeholder_call,
    is_tracing_setting_disabled,
    should_skip_tracing_for_op,
)
from weave.trace.serialization.serialize import to_json
from weave.trace.settings import (
    should_redact_pii,
)
from weave.trace_server.trace_server_interface import (
    CallEndReq,
    EndedCallSchemaForInsert,
)
import tiktoken

global llm_usage
llm_usage = {}
async def patched_anext(self):
    if not hasattr(self._iterator_or_ctx_manager, "__anext__"):
        try:
            # This is kept as a type ignore because the `google-generativeai` pkg seems
            # to yield an object that has properties of both value and iterator, but doesn't
            # seem to pass the isinstance(obj, Iterator) check...
            self._iterator_or_ctx_manager = aiter(self._iterator_or_ctx_manager)  # type: ignore
        except TypeError:
            raise TypeError(
                f"Cannot call anext on an object of type {type(self._iterator_or_ctx_manager)}"
            )
    try:
        value = await self._iterator_or_ctx_manager.__anext__()  # type: ignore
        try:
            self._on_yield(value)
            if value.usage:
                llm_usage[value.id] = value.usage
            # Here we do a try/catch because we don't want to
            # break the user process if we trip up on processing
            # the yielded value
        except Exception as e:
            # We actually use StopIteration to signal the end of the iterator
            # in some cases (like when we don't want to surface the last chunk
            # with usage info from openai integration).
            if isinstance(e, (StopAsyncIteration, StopIteration)):
                raise
            if get_raise_on_captured_errors():
                raise
    except (StopAsyncIteration, StopIteration) as e:
        self._call_on_close_once()
        raise StopAsyncIteration
    except Exception as e:
        self._call_on_error_once(e)
        raise
    else:
        return value
    
async def patched_anext_for_tiktoken(self):
    if not hasattr(self._iterator_or_ctx_manager, "__anext__"):
        try:
            # This is kept as a type ignore because the `google-generativeai` pkg seems
            # to yield an object that has properties of both value and iterator, but doesn't
            # seem to pass the isinstance(obj, Iterator) check...
            self._iterator_or_ctx_manager = aiter(self._iterator_or_ctx_manager)  # type: ignore
        except TypeError:
            raise TypeError(
                f"Cannot call anext on an object of type {type(self._iterator_or_ctx_manager)}"
            )
    try:
        value = await self._iterator_or_ctx_manager.__anext__()  # type: ignore
        try:
            self._on_yield(value)
            # if value.usage:
            #     llm_usage[value.id] = value.usage
            encoding = tiktoken.get_encoding("cl100k_base")
            if llm_usage.get(value.id) is None:
                llm_usage[value.id] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
                for msg in range(len(json.loads(self._iterator_or_ctx_manager.response.request.content.decode('utf-8'))['messages'])):
                    prompt_msg = json.loads(self._iterator_or_ctx_manager.response.request.content.decode('utf-8'))['messages'][msg]['content']
                    llm_usage[value.id]['prompt_tokens'] += len(encoding.encode(prompt_msg))
                    llm_usage[value.id]['total_tokens'] += len(encoding.encode(prompt_msg))
            for cho in value.choices:
                llm_usage[value.id]['completion_tokens'] += len(encoding.encode(cho.delta.content))
                llm_usage[value.id]['total_tokens'] += len(encoding.encode(cho.delta.content))
            
            # Here we do a try/catch because we don't want to
            # break the user process if we trip up on processing
            # the yielded value
        except Exception as e:
            # We actually use StopIteration to signal the end of the iterator
            # in some cases (like when we don't want to surface the last chunk
            # with usage info from openai integration).
            if isinstance(e, (StopAsyncIteration, StopIteration)):
                raise
            if get_raise_on_captured_errors():
                raise
    except (StopAsyncIteration, StopIteration) as e:
        self._call_on_close_once()
        raise StopAsyncIteration
    except Exception as e:
        self._call_on_error_once(e)
        raise
    else:
        return value
    

@trace_sentry.global_trace_sentry.watch()
def patched_finish_call(
    self,
    call,
    output: Any = None,
    exception: BaseException | None = None,
    *,
    op: Op | None = None,
) -> None:
    if (
        is_tracing_setting_disabled()
        or (op is not None and should_skip_tracing_for_op(op))
        or is_placeholder_call(call)
    ):
        return None

    from weave.trace.api import _global_postprocess_output

    ended_at = datetime.datetime.now(tz=datetime.timezone.utc)
    call.ended_at = ended_at
    original_output = output
    
    if op is not None and op.postprocess_output:
        postprocessed_output = op.postprocess_output(original_output)
    else:
        postprocessed_output = original_output

    if _global_postprocess_output:
        postprocessed_output = _global_postprocess_output(postprocessed_output)

    self._save_nested_objects(postprocessed_output)
    output_as_refs = map_to_refs(postprocessed_output)
    call.output = postprocessed_output
    try:
        call.output['usage'] = llm_usage[call.output['id']]
        call.output['usage'] = llm_usage[call.output['id']].dict()
    except:
        ... 
    # Summary handling
    summary = {}
    if call._children:
        summary = sum_dict_leaves([child.summary or {} for child in call._children])
    elif (
        isinstance(original_output, dict)
        and "usage" in original_output
        and "model" in original_output
    ):
        summary["usage"] = {}
        summary["usage"][original_output["model"]] = {
            "requests": 1,
            **original_output["usage"],
        }
    elif hasattr(original_output, "usage") and hasattr(original_output, "model"):
        # Handle the cases where we are emitting an object instead of a pre-serialized dict
        # In fact, this is going to become the more common case
        model = original_output.model
        usage = original_output.usage
        if isinstance(usage, pydantic.BaseModel):
            usage = usage.model_dump(exclude_unset=True)
        if isinstance(usage, dict) and isinstance(model, str):
            summary["usage"] = {}
            summary["usage"][model] = {"requests": 1, **usage}

    # JR Oct 24 - This descendants stats code has been commented out since
    # it entered the code base. A screenshot of the non-ideal UI that the
    # comment refers to is available in the description of that PR:
    # https://github.com/wandb/weave/pull/1414
    # These should probably be added under the "weave" key in the summary.
    # ---
    # Descendent error tracking disabled til we fix UI
    # Add this call's summary after logging the call, so that only
    # descendents are included in what we log
    # summary.setdefault("descendants", {}).setdefault(
    #     call.op_name, {"successes": 0, "errors": 0}
    # )["successes"] += 1
    call.summary = summary

    # Exception Handling
    exception_str: str | None = None
    if exception:
        exception_str = exception_to_json_str(exception)
        call.exception = exception_str

    project_id = self._project_id()

    # The finish handler serves as a last chance for integrations
    # to customize what gets logged for a call.
    if op is not None and op._on_finish_handler:
        op._on_finish_handler(call, original_output, exception)

    def send_end_call() -> None:
        maybe_redacted_output_as_refs = output_as_refs
        if should_redact_pii():
            from weave.trace.pii_redaction import redact_pii

            maybe_redacted_output_as_refs = redact_pii(output_as_refs)

        output_json = to_json(
            maybe_redacted_output_as_refs, project_id, self, use_dictify=False
        )
        self.server.call_end(
            CallEndReq(
                end=EndedCallSchemaForInsert(
                    project_id=project_id,
                    id=call.id,
                    ended_at=ended_at,
                    output=output_json,
                    summary=summary,
                    exception=exception_str,
                )
            )
        )

    self.future_executor.defer(send_end_call)

    call_context.pop_call(call.id)



from typing import Any, Union
from weave.trace_server_bindings.remote_http_trace_server import RemoteHTTPTraceServer, StartBatchItem, EndBatchItem, Batch, logger
import logging
import os
import json
import time

global llm_spans
llm_spans = {}
def patched_flush_calls(
    self,
    batch: list[Union[StartBatchItem, EndBatchItem]],
    *,
    _should_update_batch_size: bool = True,
) -> None:
    """Process a batch of calls, splitting if necessary and sending to the server.

    This method handles the logic of splitting batches that are too large,
    but delegates the actual server communication (with retries) to _send_batch_to_server.
    """
    # Call processor must be defined for this method
    assert self.call_processor is not None
    if len(batch) == 0:
        return

    data = Batch(batch=batch).model_dump_json()
    encoded_data = data.encode("utf-8")
    encoded_bytes = len(encoded_data)

    # Update target batch size (this allows us to have a dynamic batch size based on the size of the data being sent)
    estimated_bytes_per_item = encoded_bytes / len(batch)
    if _should_update_batch_size and estimated_bytes_per_item > 0:
        target_batch_size = int(
            self.remote_request_bytes_limit // estimated_bytes_per_item
        )
        self.call_processor.max_batch_size = max(1, target_batch_size)

    # If the batch is too big, split it in half and process each half
    if encoded_bytes > self.remote_request_bytes_limit and len(batch) > 1:
        split_idx = int(len(batch) // 2)
        self._flush_calls(batch[:split_idx], _should_update_batch_size=False)
        self._flush_calls(batch[split_idx:], _should_update_batch_size=False)
        return

    # If a single item is too large, we can't send it -- log an error and drop it
    if encoded_bytes > self.remote_request_bytes_limit and len(batch) == 1:
        logger.error(
            f"Single call size ({encoded_bytes} bytes) is too large to send. "
            f"The maximum size is {self.remote_request_bytes_limit} bytes."
        )

    try:
        # self._send_batch_to_server(encoded_data)
        # sorted_batch = sorted(batch, key=lambda x: (x.mode != 'start',))
        for b in batch:
            if b.req.dict().get('start') is not None:
                # file_path = './results/' + b.req.dict()['start']['project_id'] + '/' +b.req.dict()['start']['id'] + '.json'
                # os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # with open(file_path,'w') as f:
                #     json.dump(json.loads(data), f)
                start_time_stamp = b.req.start.started_at.timestamp()
                id = b.req.start.id
                is_omni_run_trace = ('omni_run' in b.req.start.op_name)
                logging.info(f'LLM completion start, id:{id}, timestamp: {start_time_stamp}, is_omni_run_trace: {is_omni_run_trace}, op_name: {b.req.start.op_name}')
            if b.req.dict().get('end') is not None:
                # model_name = b.req.end.output['model']
                if b.req.end.summary.get('usage') is not None:
                    for k in b.req.end.summary['usage']:
                        prompt_tokens = b.req.end.summary['usage'][k]["prompt_tokens"]
                        completion_tokens = b.req.end.summary['usage'][k]["completion_tokens"]
                        total_tokens = b.req.end.summary['usage'][k]["total_tokens"]
                        end_time_stamp = b.req.end.ended_at.timestamp()
                        logging.info(f'LLM name: {k}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}, id: {b.req.end.id}, timestamp: {end_time_stamp}')
        # file_name = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
        # file_save_path = os.path.join(result_path, file_name)
        # with open(file_save_path,'w') as f:
        #     json.dump(json.loads(data), f) 
    except Exception:
        # Add items back to the queue for later processing
        logger.warning(
            f"Batch failed after max retries, requeueing batch with {len(batch)=} for later processing",
        )

        # only if debug mode
        if logger.isEnabledFor(logging.DEBUG):
            ids = []
            for item in batch:
                if isinstance(item, StartBatchItem):
                    ids.append(f"{item.req.start.id}-start")
                elif isinstance(item, EndBatchItem):
                    ids.append(f"{item.req.end.id}-end")
            logger.debug(f"Requeueing batch with {ids=}")
        self.call_processor.enqueue(batch)

def init_token_count(offline_mode=False, tiktoken=False):
    if tiktoken:
        _IteratorWrapper.__anext__ = patched_anext_for_tiktoken
    else:
        _IteratorWrapper.__anext__ = patched_anext
    WeaveClient.finish_call = patched_finish_call
    if offline_mode:
        RemoteHTTPTraceServer._flush_calls = patched_flush_calls