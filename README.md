# 🪄 Statecraft -  Store, manage and remix states for SSMs and Stateful models

## TL;DR

With statecraft you can easily load states from a file or from the community repository and use them in your SSMs or stateful models. The interface is just like Huggingface's Transformers library:

```python
from statecraft import StatefulModel

model = StatefulModel.from_pretrained(
        model_name="state-spaces/mamba-130m-hf",
        initial_state_name="koayon/state-a",
    )
```

Now with every forward pass you get the knowledge from the state you loaded. And because StatefulModel inherits from Huggingface's `PreTrainedModel`, you can use it just like any other Huggingface model.

## 🧑‍💻 Installation

```bash
pip install statecraft
```

## ✨ Other Features

- Use `model.build_state()` to generate a new state from your context.
- Use `model.save_state()` or `model.save_current_state` to save your state to a file so that you can use it again in the future.
- With `model.load_state()` you can load a state either from a file or from the community repository.

## 🔍 Coming Soon

- Right now we only support Mamba models (in all sizes), as more SSMs and Stateful models begin to come onto Huggingface, we will support them too.
- The statecraft community repository is available right now and a front-end for this is coming so you can view all of the community-contributed states. Right now, you can see the available states by running `statecraft.show_available_states()`.
- We're also looking at RAG-like generation approaches where you automatically retrieve the `state` instead of `context`, watch this space 👀

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
