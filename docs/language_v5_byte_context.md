# Language v5: byte + contextual teacher path

Status: EXPERIMENTAL / LOCAL_SMOKE_VERIFIED.

This change addresses the live HNT2 collapse where a correct phrase such as
`Я это знаю.` could become the dominant answer even when the teacher context
changed.

The previous live path had two important limits:

1. `learn_teacher_text()` trained the expressive model under a learned null
   context. It taught surface language but not `teacher context -> response`.
2. correction learning used an already detached 96-D receptive vector, so the
   receptive encoder did not learn the contextual distinction from the same
   expression loss.

v5 keeps the scalable StateWeave hybrid but restores exact UTF-8 byte units
(260 IDs including PAD/BOS/EOS/UNK) and adds
`learn_conditioned_expression(context, target)`. That method keeps the
receptive encoder and expressive decoder in one autograd graph.

HNT3 remains live/dynamic. The teacher returns compact JSON:

`{"say":"...","teach":["..."],"fix":[]}`

`teach` is explicit teacher demonstration data for the current `say`; it is
not substituted for Heiwa's own answer. `fix` remains restricted to a real
correction/rephrasing of Heiwa's immediately preceding utterance. Heiwa's own
text is never used as teacher data.

The v5 byte checkpoint is intentionally separate:
`state/language_developmental_dialogue.v5-byte.pt`.
Old v4 checkpoints are not silently loaded into the byte vocabulary.
