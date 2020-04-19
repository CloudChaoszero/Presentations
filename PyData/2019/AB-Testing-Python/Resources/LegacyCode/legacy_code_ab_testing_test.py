# x1 = np.linspace(-4,20,100)
# mean1 = np.mean(x1)
# std1 = np.std(x1)

# y1 = scipy.stats.norm.pdf(x1,mean1,std1)


# x2 = np.linspace(6,30,100)
# mean2 = np.mean(x2)
# std2 = np.std(x2)

# y2 = scipy.stats.norm.pdf(x2,mean2,std2)
# deltaX = np.linspace(min(x2), z, 100)
# deltaY = y2-y1

# norm = scipy.stats.norm


# pepe_params = norm.fit(x1)
# modern_params = norm.fit(x2)

# xmin = min(x1.min(), x2.min())
# xmax = max(x1.max(), x2.max())
# x = np.linspace(-15, 35, 100)

# pepe_pdf = norm(*pepe_params).pdf(x)
# modern_pdf = norm(*modern_params).pdf(x)
# y = np.minimum(modern_pdf, pepe_pdf)

# fig, ax = plt.subplots()
# ax.set_ylim(min(modern_pdf),max(modern_pdf))
# ax.axvline(1.96 * (np.std(x)/(np.sqrt(100)-1)) + np.mean(x) , color='red')
# ax.plot(x, pepe_pdf, color='blue')
# ax.plot(x, modern_pdf, color='orange')
# ax.fill_between(x, y, color='red', alpha=0.3)
# plt.show()