# Solution
training = employment.loc[:"2000-01-01"]
testing = employment.loc["2000-01-02":]

statistics = stattools.arma_order_select_ic(training)  # May take a few minutes

statistics['bic']

p, q = statistics.bic_min_order

model = sms.tsa.ARIMA(training, order=(p,0, q))

results = model.fit()

results.summary()

y_pred = results.predict(start=testing.index.min(), end=testing.index.max())

plt.plot(y_pred, 'r--')
plt.plot(testing)
plt.show()