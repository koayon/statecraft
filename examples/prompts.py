abstract = """Foundation models, now powering most of the exciting applications in deep learning, are almost universally
based on the Transformer architecture and its core attention module. Many subquadratic-time architectures
such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs)
have been developed to address Transformers’ computational inefficiency on long sequences, but they have not
performed as well as attention on important modalities such as language. We identify that a key weakness of
such models is their inability to perform content-based reasoning, and make several improvements. First, simply
letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing
the model to selectively propagate or forget information along the sequence length dimension depending on
the current token. Second, even though this change prevents the use of efficient convolutions, we design a
hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified
end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast
inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance
improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves
state-of-the-art performance across several modalities such as language, audio, and genomics. On language
modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice
its size, both in pretraining and downstream evaluation."""

MAMBA_ABSTRACT = abstract.replace("\n", " ")

wiki = """Mambas are fast-moving, highly venomous snakes of the genus Dendroaspis (which literally
means "tree asp") in the family Elapidae. Four extant species are recognised currently; three of those four
species are essentially arboreal and green in colour, whereas the black mamba, Dendroaspis polylepis, is largely
terrestrial and generally brown or grey in colour. All are native to various regions in sub-Saharan Africa and all
are feared throughout their ranges, especially the black mamba. In Africa there are many legends and stories about mambas.

The three green species of mambas are arboreal, whereas the black mamba is largely terrestrial. All four species are active
diurnal hunters, preying on birds, lizards, and small mammals. At nightfall some species, especially the terrestrial black mamba,
shelter in a lair. A mamba may retain the same lair for years. Resembling a cobra, the threat display of a mamba includes rearing,
opening the mouth and hissing. The black mamba's mouth is black within, which renders the threat more conspicuous. A rearing mamba
has a narrower yet longer hood and tends to lean well forward, instead of standing erect as a cobra does."""

wiki_2 = """Stories of black mambas that chase and attack humans are common, but in fact the snakes generally avoid contact with humans.
The black mamba (Dendroaspis polylepis) is a highly venomous snake species native to various parts of sub-Saharan Africa.
Black mambas are fast-moving, nervous snakes that will strike when threatened.
According to findings by Branch (2016), their venom comprises neurotoxins and cardiotoxins that can rapidly induce symptoms, including dizziness,
extreme fatigue, vision problems, foaming at the mouth, paralysis, convulsions, and eventual death from respiratory or cardiac failure if untreated.
Although black mamba venom is highly toxic, antivenom is available and can treat envenomation promptly."""

MAMBA_WIKI = wiki.replace("\n", " ")
