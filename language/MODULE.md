# LANGUAGE MODULE

## PURPOSE

Language is Heiwa's trainable linguistic organ: receptive representation, explicitly supervised linguistic learning, surface competence and expression realization.

Language is not Policy, personality, motivation or autobiography.

## INPUT

- external human/mentor text;
- explicitly marked teacher text;
- structured lexical/compositional supervision;
- later: stable semantic/Core state for expression.

## OUTPUT

- receptive language representation;
- learned linguistic competence;
- generated text when cognition has selected a linguistic action.

## OWNS STATE

May own:
- Unicode character embeddings/tokenizer;
- receptive weights;
- expressive decoder weights;
- optimizer state;
- language telemetry;
- structured curricula/checkpoints.

It does not own global goals/personality/World state.

## RULES

1. Ordinary dialogue is not teacher evidence.
2. Own generated speech is not teacher evidence.
3. Explicit teacher data may update Language.
4. Language must not decide when to speak/ask.
5. No canned whole-phrase reply table.

## TOKENIZATION

Architecture v3 uses one serialized Unicode character vocabulary for both
receptive and expressive paths. It includes PAD/BOS/EOS/UNK, Russian and Latin
upper/lower case, digits, whitespace and basic punctuation. One Russian letter
is one token; unknown Unicode becomes visible `<UNK>` rather than invalid UTF-8.

## CURRENT DEVELOPMENT

Current target:

`atoms -> reusable roles/slots -> compositions -> Russian surface forms -> heldout novel compositions`

Current curriculum is approximately:
- 18 atoms;
- 24 training compositions;
- 8 heldout compositions.

Heldout data must not enter gradient updates.

## VERIFIED FAILURE TO AVOID

V1 created fresh random vectors for bindings/whole compositions. Heldout combinations therefore contained unseen random targets even when their atoms were known.

Observed V1 retrieval stayed at random baseline.

Never restore that design.

## CURRENT V3

Status: LOCAL_PATCH_VERIFIED (no live mentor run).

`language_arch_version = 3` keeps the Unicode/API/checkpoint line and adds a
Heiwa-native `StateWeave` temporal branch to every hybrid block. StateWeave
contains fast/medium/slow persistent states, input-conditioned low-rank state
transition corrections, and a small delta-rule associative fast-weight memory.
It is an actual trainable PyTorch recurrent architecture, not a diagram and not
a Mamba-2 claim.

Each hybrid block now combines:

`StateWeave + SSM backend + gated recurrent memory + local attention -> learned fusion -> FFN`.

The existing `selective_ssm` remains a secondary portable baseline/fallback. If
an official `mamba_ssm.Mamba2` backend later passes the isolated local probe it
can occupy the SSM branch without replacing StateWeave or the public Language
API.

The local profile is approximately 166M trainable parameters. Width, depth,
attention window and StateWeave ranks can scale without changing the public
Language interface.

Semantic targets remain entirely composed from reusable concept and role/slot
components. No fresh random sentence vector is introduced.

Surface LM occurs before semantic grounding in the epoch.

Core-conditioned expression is paused until Core conditioning is persistent/stable.

## TRAINING SOURCES

Allowed:
- explicit human/mentor teacher data;
- structured explicit curriculum.

Forbidden automatic sources:
- Heiwa's own speech;
- ordinary unmarked conversation;
- arbitrary raw ChatGPT transcript.

## PERSISTENCE

Historical:
- `state/language_bootstrap.pt`
- `state/language_compositional.pt`

Current expected V3:
- `state/language_compositional_v3.pt`

V3 checkpoints contain architecture version, StateWeave version, exact resolved SSM backend,
config, tokenizer/vocabulary, model state, optimizer state and training
metadata. V1 byte/GRU checkpoints are reported incompatible, retained intact,
and new developmental saves use an explicit `.v3.pt` sibling.

Curriculum:
- `state/language_curriculum/compositional.json`

## TELEMETRY

Watch:
- semantic/receptive loss;
- surface loss;
- atom_top1;
- train retrieval/top1;
- holdout_top1;
- train/holdout cosine;
- minimal contrast margin.

Random current baselines:
- atom ~0.056
- train ~0.042
- holdout 0.125

Falling loss without improved retrieval/contrast is not enough.

## MUST NOT

- become Policy;
- use self-output as teacher;
- use ordinary dialogue as automatic teacher;
- implement phrase lookup;
- create a fresh random Core conditioning space each run;
- train on holdout.

## NEXT

1. Verify V3/StateWeave on the actual GPU through the existing developmental loop.
2. Build persistent Language<->Core semantic bridge.
3. Train expression from that stable space.
4. Test novel compositions.
5. Then use language for autonomous inquiry.

## READ SCOPE

For current work:
- `AGENTS.md`
- `AI_HANDOFF.md`
- `AI_STATE.md`
- `language/MODULE.md`
- `language/interface.py` if needed
- `language/compositional_pedagogy.py`
- `runtime/language_compositional_school.py`

Open Core internals only when implementing the persistent bridge.

## LIVE DEVELOPMENTAL DIALOGUE

Immediate active path is no longer repeated offline epoch training alone.

`language/developmental.py` provides teacher-only developmental mechanics:
- deterministic semantic curriculum target;
- own Language realization attempt;
- explicit mentor correction parsing/training support;
- retry from the same semantic target.

The target scaffold is NOT runtime desire/Policy. It exists only so early language can be taught before the persistent Core-language bridge is ready.

Own failed attempts are never teacher evidence. Mentor correction forms are explicit teacher evidence.
