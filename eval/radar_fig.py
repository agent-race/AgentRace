import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
# print(plt.style.available)

plt.style.use('seaborn-v0_8-muted')

plt.rcParams['font.serif']=['Times New Roman']


font = FontProperties(size=6)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors.append('#E1821A')

# 使用matplotlib作图
fig = plt.figure()
fig.set_facecolor('#FFFFFF')
fig.subplots_adjust(wspace=0.5, hspace=0.20, top=0.85, bottom=0.05)

#图1

algs=['LangChain','AutoGen','AgentScope','CrewAI','LlamaIndex','Phidata', 'PydanticAI']

# stats_ls = [
#     [70.00  , 62.50  , 56.73  , 54.80  , 46.71  , 67.57  , 46.83  , 43.33  , 41.08  , 45.7  , 48.96  , 46.85],
#     [65.00  , 65.91  , 58.65  , 51.41  , 46.05  , 56.76  , 39.02  , 39.33  , 34.85  , 35.40  , 35.52  , 29.72],
#     [46.00  , 43.18  , 38.46  , 33.90  , 32.89  , 32.43  , 27.32  , 24.67  , 28.63  , 19.59  , 27.76  , 23.78],
#     [60.00  , 76.14  , 72.12  , 69.49  , 57.24  , 71.17  , 64.39  , 58.67  , 52.70  , 66.67  , 68.06  , 67.83],
#     [69.00  , 64.77  , 52.88  , 41.24  , 46.05  , 55.86  , 38.54  , 41.33  , 39.83  , 33.33  , 33.73  , 34.97],
#     [80.00  , 84.09  , 75.00  , 81.92  , 69.74 , 72.07  , 68.78  , 61.33  , 56.85  , 52.23  , 66.57  , 59.44],
#     [80.00  , 84.09  , 75.00  , 81.92  , 69.74 , 72.07  , 68.78  , 61.33  , 56.85  , 52.23  , 66.57  , 59.44]]

stats_ls = [[ 0.02423455, 0.00003333, 0.06422606, 0.0004194, 0.00009758, 0, 0.5345976, 0, 0.0152988,1.58856],
[0.0009297,0.000336,0.002387,0.00002909,0.0002212, 0, 1.05489,0,	0.00005333,9.4219,],
[0.217,0.000297,0.00405,0.0000193,0.00000883,0.729,4.083,0.0000271,0.752,7.291,],
[0.00965,0.000196,0.00422,0.00123,0.000278,0.000346,0.03164,0.000999,0.09565,4.031,],
[0.0001352,0.00016616,0.004254,0.000034839,0.0001135,0.03341,0.8767,0,0.05782,1.4399,],
[0.001147,0.0007207,0.003858,0.0002107,0.000073355,0.03821,1.4065,1.38445E-05,0.003035,1.83012,],
[0.001395,0.0003148,0.003795,8.6865E-06,0.000056241,0.02965,1.2104,3.1952E-06,0.0001414,1.2275,]]

# labels=['Level 1','Level 2','Level 3','Level 4','Level 5','Level 6','Level 7','Level 8','Level 9','Level 10','Level 11','Level 12']
labels=['PDF tool','CSV tool','Xlsx tool','Txt tool','Docx tool','Audio tool','Vision tool','Video tool','Python tool','Web browser\n tool']

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
# Create figure
fig = plt.figure()
fig.set_facecolor('#FFFFFF')
fig.subplots_adjust(wspace=0.5, hspace=0.20, top=0.5, bottom=0.05)

# Create polar subplot with log scale
ax = fig.add_subplot(121, polar=True)
ax.set_yscale('log')  # Set radial axis to logarithmic scale

# Set custom grid lines and labels for log scale
ax.set_rgrids([0.0001, 0.001, 0.01, 0.1, 1, 10], 
              angle=0, 
              fontproperties=FontProperties(size=6),
              fontsize=4,
              color='grey')

# 自定义格式化函数
def sci_notation(y, _):
    if y == 0:
        return "0"
    exponent = int(np.log10(y))
    coeff = y / 10**exponent
    return f"10$^{{{exponent}}}$"  
    # return f"{coeff:.0f}×10$^{{{exponent}}}$"  


# Format the radial axis labels to show exponents
ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation))

lines = []
for i, stats in enumerate(stats_ls):
    stats = np.concatenate((stats, [stats[0]]))
    line, = ax.plot(np.concatenate((angles, [angles[0]])), stats, 
                    linewidth=0.75, color=colors[i], label=algs[i])
    lines.append(line)
    ax.fill(np.concatenate((angles, [angles[0]])), stats, 
            alpha=0.1, color=colors[i])

ax.set_thetagrids(angles * 180 / np.pi, labels, 
                 fontproperties=font, fontname='Times New Roman')
ax.spines['polar'].set_visible(False)
# ax.set_title("(a) Closed-source LMMs", fontsize=8, loc="center", y=-0.25, fontname='Times New Roman')

ax.legend(handles=lines, loc='upper center', ncol=2, 
          bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=False, 
          frameon=True, prop=FontProperties(size=6, family='Times New Roman'), 
          framealpha=0.1)

plt.savefig('radar_all_level_log.pdf', format='pdf', 
            dpi=500, bbox_inches='tight', pad_inches=+0.1)
# plt.show()