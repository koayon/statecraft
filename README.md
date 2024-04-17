# 🪄 Statecraft -  Load, store and remix states for SSMs, Mamba and Stateful models

## TL;DR

With statecraft you can easily load states from a file or from the [community repository](https://www.statecrafthub.com/) and use them in your SSMs or stateful models. The interface is just like Huggingface's Transformers library:

```python
from statecraft import StatefulModel

model = StatefulModel.from_pretrained(
        model_name="state-spaces/mamba-130m-hf",
        initial_state_name="koayon/state-a",
    )
```

Now with every forward pass you get the knowledge from the state you loaded. And because StatefulModel inherits from Huggingface's `PreTrainedModel`, you can use it just like any other Huggingface model.

## Quick Start

Here's an example of loading a state built from the start of the wikipedia page for the Mamba snake and asking the model to add a summary:

```python
model = StatefulModel.from_pretrained(
    model_name="state-spaces/mamba-2.8b-hf",
    initial_state_name="jeremy1/mamba_wiki_state",
)

tokeniser = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
prompt = "In summary,"
input_ids = tokeniser(prompt, return_tensors="pt")["input_ids"]

output_tokens = model.generate(input_ids=input_ids, do_sample=False)
print(tokeniser.decode(output_tokens[0]))
```

returns:

```text
In summary, the black mamba is the most venomous snake in the world, with
a venom that is more than 100 times more toxic than that of the cobra.
```

Note that our prompt contains nothing about snakes, all this information comes from the state we loaded.

## 🧑‍💻 Installation

```bash
pip install statecraft
```

## ✨ Other Features

- Use `model.build_state()` to generate a new state from your context.
- Use `model.save_state()` or `model.save_current_state` to save your state to a file so that you can use it again in the future.
- With `model.load_state()` you can load a state either from a file or from the community repository.
  - To see the states that are available in the community repository, visit the [Statecraft Hub](https://statecrafthub.com) or use `statecraft.list_states()`.

## 🔍 Coming Soon

- Right now we only support Mamba models (in all sizes), as more SSMs and Stateful models begin to come onto Huggingface, we will support them too.
- We're also looking at RAG-like generation approaches where you automatically retrieve the `state` instead of `context`, watch this space 👀

## ❔ What Is A Stateful Model?

A Stateful Model is a class of models that have a finite state which contains information from previous tokens in the context window. SSMs are the paradigmatic example but hybrid models also fall into this category.

Examples of Stateful Models include:

- [Mamba](https://huggingface.co/docs/transformers/en/model_doc/mamba)
- (Striped) [Hyena](https://www.together.ai/blog/stripedhyena-7b)
- Mamba hybrids like [Jamba](https://www.ai21.com/jamba) and [Zamba](https://www.zyphra.com/zamba)
- [RecurrentGemma](https://huggingface.co/docs/transformers/main/en/model_doc/recurrent_gemma) ([Griffin](https://arxiv.org/abs/2402.19427v1))

## 🧙 Conceptually

Currently, we often use RAG to give a transformer contextual information.

With Mamba-like models, you could instead imagine having a library of states created by running the model over specialised data. States could be shared kinda like LoRAs for image models.

For example, I could do inference on 20 physics textbooks and, say, 100 physics questions and answers. Then I have a state which I can give to you. Now you don’t need to add any few-shot examples; you just simply ask your question. The in-context learning is in the state.

In other words, you can drag and drop downloaded states into your model, like literal plug-in cartridges. And note that “training” a state doesn’t require any backprop. It’s more like a highly specialised one-pass fixed-size compression algorithm. This is unlimited in-context learning applied at inference time for zero-compute or latency.

The structure of an effective Transformer LLM call is:

- System Prompt
- Preamble
- Few shot-examples
- Question

With statecraft and your SSM, we instead simply have:

- Inputted state (with problem context, initial instructions, textbooks, and few-shot examples all baked in)
- Short question

For more details see [here](https://www.kolaayonrinde.com/blog/2024/02/11/mamba.html#:~:text=Swapping%20States%20as%20a%20New%20Prompting%20Paradigm)

## 💃 Contributions

We welcome contributions to the statecraft library!
And of course, please contribute your SSM states with `statecraft.upload_state(...)` 🪄

---

Proudly powering the SSM revolution.
