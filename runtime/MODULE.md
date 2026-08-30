# Runtime

## PURPOSE

Оркестрация одного causal cognitive cycle.

Runtime — не cognition.

## FLOW

Observation
→ injected CoreInputBuilder
→ Core
→ Policy
→ ActionIntent
→ ActionAttempt
→ World executor
→ delayed CompletedAction OR failure Consequence
→ injected ConsequenceInterpreter
→ transition-specific Core/Policy trace

## OWNS STATE

- current CoreState reference;
- transition numbering/bookkeeping.
- pending causal transitions until a terminal Consequence arrives.

## MUST NOT

Runtime не должен:

- выбирать действие вместо Policy;
- превращать отказ в CompletedAction;
- создавать domain strategy;
- подменять World-specific execution.

## CAUSAL GUARANTEE

Все элементы action chain должны сохранять исходный transition_id.

Rejected/failed attempt не создаёт CompletedAction.

Runtime принимает learning signal только для сохранённого pending transition.
Если terminal Consequence не даёт credit конкретному модулю, trace этого
модуля удаляется, а не обучается от чужого события.

Перед cross-organ learning Runtime проверяет наличие всех traces, которым
назначен credit. Если хотя бы одна trace уже вытеснена, ни один орган не
обновляется: оставшиеся traces удаляются, terminal transition закрывается с
явной причиной `missing_causal_trace`, и следующий transition продолжает loop.

## COORDINATED PERSISTENCE

`CausalCognitionCycle.save_checkpoint(path)` записывает один versioned snapshot:

- Runtime transition prefix/counter и точные pending lifecycle records;
- уже полученный `CompletedAction`, даже если `Consequence` ещё задержан;
- opaque Core/Policy snapshots с тем же generation id и точным набором traces;
- SHA-256 каждого organ blob.

Файл сначала полностью пишется рядом с целью и публикуется через atomic replace.
Restore разрешён только в свежий cycle с тем же World/prefix. Envelope, hashes и
pending chain проверяются до organ restore. После любой ошибки restore cycle
poisoned: `step` и повторный save запрещены, поэтому частично восстановленный
organ не может участвовать в cognition. Перед coordinated restore Runtime также
сохраняет private rollback generation свежих Core/Policy. Если второй organ
отклоняет blob, оба organ возвращаются к точному pre-restore состоянию; poison
остаётся, даже когда rollback успешен.

Это обеспечивает продолжение delayed learning из сохранённой generation. Это
не является exactly-once transaction между checkpoint и внешним World: crash
после learning, но до следующего snapshot/World acknowledgement требует
отдельного durable acknowledgement protocol.

## WHAT DOES IT CAUSE?

Runtime обеспечивает техническую причинную связь между cognition,
исполнением и learning.

## WHAT TRAINS IT?

Runtime не обучается.

## FILES

causal.py
    World-agnostic online causal loop. Observation interpretation and
    consequence interpretation are injected interfaces; Runtime не содержит
    domain rules.

causal_smoke.py
    CPU-only controlled end-to-end scenario: persistent Core + dynamic Policy
    + simulated World + delayed causal consequence + changed next decision.

## READ SCOPE

Для изменения Runtime обычно:

- MODULE.md
- causal.py
- public contracts
- Core/Policy interfaces

## ACTIVE ACTION LIFECYCLE

Некоторые actions не завершаются внутри одного cognitive transition.

Например:

- удержание клавиши;
- drag;
- длительный scroll;
- внешняя операция, которая продолжает выполняться.

Правильная модель:

ActionIntent
→ ActionAttempt
→ ActiveAction

После этого World должен иметь возможность предоставить новые
Affordances:

- continue active action;
- cancel active action;
- inspect active action.

Policy решает, какой Affordance выбрать.

Runtime только хранит causal lifecycle.

Отмена является НОВЫМ cognitive transition:

original transition
→ ActiveAction

later transition
→ cancel intent
→ cancel attempt
→ execution
→ CancelledAction

Нельзя использовать cognitive костыли:

if repeated_action:
    cancel()

if duration > N:
    choose_other_action()

Аварийный safety watchdog в будущем допустим только как техническая
защита исполнения, а не как cognitive decision.
