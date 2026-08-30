# Market World

## STATUS

PARKED EXPERIMENT.

Market НЕ является текущим первым World.

Текущий development direction — Desktop World.

Ранее созданный local Market simulation можно сохранить как проверку
универсальности архитектуры, но сейчас его не расширять.

Не добавлять:

- T-Bank;
- реальные orders;
- новые trading models;
- Market-specific learning;
- trading strategy;

без явного решения вернуться к Market.

## ARCHITECTURAL ROLE

В будущем Market должен быть ещё одним World той же persistent Heiwa:

Market
→ Observation/Affordances
→ общий Core/Policy
→ ActionIntent
→ execution
→ Consequence
→ общий Learning

Market не должен становиться отдельным мозгом Heiwa.

## READ SCOPE

Не читать этот module при разработке Desktop.
