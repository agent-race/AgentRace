from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
from functools import wraps

def traced_tool_dec(tool_name=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs): 
            span = tracer.start_span(fn.__name__)
            span.set_attribute("input", str(args))
            try:
                result = fn(*args, **kwargs)
                span.set_attribute("output", str(result)[:100])
                return result
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                span.end()
                name = tool_name if tool_name else fn.__name__
                print(f"{name} finished, time: {(span.end_time - span.start_time)/1e9}")
        return wrapper
    return decorator

def traced_tool(fn, tool_name=None):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        span = tracer.start_span(fn.__name__)
        span.set_attribute("input", str(args))
        try:
            result = fn(*args, **kwargs)
            span.set_attribute("output", str(result)[:100])  # 记录截断的输出
            return result
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            if tool_name:
                name = tool_name
                print(f"{name} finished, time: {span.end_time - span.start_time}")
            else:
                print(f"{fn.__name__} finished, time: {(span.end_time - span.start_time)/1e9}")
    return wrapper