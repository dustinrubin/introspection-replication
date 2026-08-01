# Introspection Replication

Fork of [victorgodet/llm-experiments](https://github.com/victorgodet/llm-experiments) with additional control experiments for the [introspection via localization](https://www.lesswrong.com/posts/3HXAQEK86Bsbvh4ne) protocol.

The original experiment tests whether an LLM can localize which sentence a steering vector was injected into, framed as introspection. I ran the same protocol with alternative prompts to test what the model is actually detecting.

## Control results

Qwen 2.5 14B, 5 sentences, 100 trials each:

| Prompt                  | Accuracy |
|-------------------------|----------|
| introspection           | 89.2%    |
| which is most abstract? | 90.0%    |
| which stands out?       | 80.4%    |
| which is most concrete? | 1.0%     |
| which do you prefer?    | 4.6%     |

The steering vectors in `prompts.txt` are specific→generic pairs (dog→animal, fire→light, etc.), which may encode "abstractness." "Abstract" matched or exceeded the introspection prompt on this and other models, suggesting the task may not require introspective framing at all.

Discussion: [comment thread on the original post](https://www.lesswrong.com/posts/3HXAQEK86Bsbvh4ne?commentId=WSQpBkN9xGowrRfBr)

## Files

- `introspection_localize.py` — prompt variants tested
- `modal_results.csv` — full experiment results
