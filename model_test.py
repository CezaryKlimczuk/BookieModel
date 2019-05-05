import data_grab as dg
import matplotlib.pyplot as plt

bookies = []
model = []
random = []
optimal = []
years = []
for year in range(6, 18):
    likelihoods = dg.compare_model(year, 'E0')
    bookies.append(likelihoods[0])
    model.append(likelihoods[1])
    random.append(likelihoods[2])
    optimal.append(likelihoods[3])
    years.append(f'{2000+year}')
    print(year)

plt.plot(years, bookies, label='bookies')
plt.plot(model, label='model')
plt.plot(random, label='random')
plt.plot(optimal, label='optimal')
plt.title('Premier League seasons')
plt.legend()
plt.grid()
plt.show()


