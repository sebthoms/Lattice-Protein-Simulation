import frontend as fr

config = fr.Configuration('config.toml')

fr.fixed_plot(config, 1, 10000)
fr.var_plot(config, 5, 1, 20, 10000)
fr.var_plot_avg(config, 10, 1, 25, 100000)

