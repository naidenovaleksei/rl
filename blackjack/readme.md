## Игра в Блекджек на основе окружения Blackjack-v0 из OpenAI Gym

Здесь реализованы алгоритмы обучения без модели: 
- Алгоритм Monte Carlo control with exploring starts
- Алгоритм on-policy Monte Carlo control с мягкими стратегиями
- Алгоритм off-policy Monte Carlo control
- Алгоритм Q-learning (off-policy TD control)

Алгоритмы испытывались:
- на обычном окружении Blackjack-v0 из OpenAI Gym
- на окружении Blackjack-v0 с добавлением действия "удвоение ставки"
- на окружении Blackjack-v0 с добавлением действия "удвоение ставки" и упрощенным подсчетом карт на основе системы "Плюс-минус"