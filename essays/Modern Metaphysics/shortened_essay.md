# Patterns, Pragmatism, and Mild Realism: An Empirical Probe

**Florin Cojocariu**  
04.05.2025

## Abstract

This essay revisits the question of whether patterns in language are real, by comparing Daniel Dennett's "mild-realism" claim that patterns are real when they help us compress and predict information, with Norton Nelkin's instrumentalist view that patterns are interpretive constructs, not features of the world itself. I examine how words like *cat*, *hammer*, or *water* create different _patterns of use[^1]_ when employed in different kinds of sentences.

By analyzing series of naturally occurring sentences with a given word by using sentence-embedding models, I show that meaning forms around a word not as a fixed point but as a shape—with a dense, literal "core" of usage and a more variable, figurative "halo." This structure emerges from the language itself, not from theoretical imposition, similarly to how a mountain is a real gradient in height, not an artifact of our measurement method. 

My approach uses SBERT to classify naturally occurring sentences into literal (P-type) and idiomatic (Q-type) uses, mapping them into semantic space using role-based projections. The resulting sentence embeddings consistently reveal a distinctive shape: a dense, stable P-core surrounded by a looser Q-cap. I argue that P-cores satisfy Dennett's criteria for real patterns—they are stable, reproducible, and predictive—while Q-caps exhibit the variability that motivates Nelkin's skepticism but are nevertheless also "real patterns;" in the end, the thesis here is that empirical evidence validates Dennett's view while it partly contradicts Nelkin's skepticism.

Philosophically, this discovery of a spectrum from literal, sensory uses to idiomatic uses suggests a solution to ancient puzzles about how words "hook" into reality, showing that concepts are not atomic objects but fluid patterns with stable cores and shifting halos.

## Introduction

What makes a pattern real? This question sits at the core of Daniel Dennett's influential 1991 paper *Real Patterns*, which argues that a pattern is real if it enables compression and predictive power—if it "pays for the cost of its own description." Norton Nelkin (1994) counters that patterns, especially in scientific or linguistic domains, are epistemic conveniences, not ontological commitments: they reflect how we organize the world, not how the world is.

I use their debate as a starting point for studying patterns in LLM sentence embeddings. The reasoning is founded on the idea that words are acquired first as simple labels for patterns of sensations. "Cat" is, at first, just a label for a shape, a texture, some specific sounds. I call this primitive label-usage the "object-word."

My method may seem an unusual one, as it blends philosophical work with concrete empirical observations on the internal structure of a Large Language Model (SBERT)	. It is my view that LLMs are exceptional research objects of a new kind, offering us a new testing ground for the philosophy of language. In this case, we can literally see how (and why) Wittgenstein's "meaning is use" is true.

I target first multiple sentences containing specific object-words like _cat_, _fork_, _tomato_, the assumption being that their distribution in the embedding space will reflect through certain patterns the way we use the words. For each word, I build then a corpus of naturally occurring sentences and classify them into P-type (literal, concrete use) and Q-type (figurative, idiomatic use). Projected into 2D space, the embeddings reveal a consistent topological form: a dense stem-like cluster of literal sentences (the rod), surrounded by a diffuse cap of idiomatic uses (the mushroom). A couple of additional dimensions can be identified and make a projection in 3D where the patterns that certain uses of the word emerge more clearly.

An extensive description of the method and links to relevant scripts used to analyze the data in SBERT can be found in the long form essay.

## From Words to Space: Semantic Embeddings and Role Axes

The empirical approach hypothesized at first five cardinal directions that track familiar philosophical contrasts—"semantic axes": Agent ↔ Object, Literal ↔ Metaphoric, Perceived ↔ Symbolic, Quantity ↔ Quality, Thing ↔ Concept. Every sentence can be located by a five-number "role vector," revealing its blend of agency, literality, and so on. We classified manually some test sentences with "cat" and we looked at their projection patterns along these axes.

As our axes are arbitrary at first, they simply serve as an initial reference system for projection but it is important to note that once the reference set, composite real axes that maximize the gradient of the distribution can be identified.

One such composite axis—angled toward both Thing vs. Sign and Literal vs. Metaphoric—seems to capture the main corridor along which meaning slides from concrete usage into idiom and displayed a clear emerging pattern. I call this the "real axis" because it captures a familiar philosophical journey: from showing to saying, from concrete to concept.

To test the idea, I constructed two contrasting sentence sets:
- **Proust Set**: sensory, phenomenal uses (e.g. "the candle flickered")  
- **Quine Set**: abstract, symbolic uses (e.g. "the candle represents hope")

The separation proved clear and measurable and, furthermore, it worked on arbitrary selections of sentences in the two categories, not used before for training the validation model. Phenomenological and analytic sentences occupy genuinely different regions of sentence space, not just different moods of the same discourse.

## Our Model: Rods and Caps

While certain precautions were employed to make sure we're not finding artifacts generated by the method, more extensive testing (using different models) is needed to full validate the results. However, we've seen how, across many examples, literal sentences huddle tightly around a center, like a slender rod[^2] and figurative sentences spread outward in wide, overlapping caps. The pattern seemed robust: rods stay compact, caps diffuse, and led us to proposing a model for this type of pattern:

**Rods (P-sentences)**: Embeddings of literal usages cluster tightly near the word's core meaning, forming a small, centered "rod" with high pairwise cosine similarities and low variance—highly compressible and repeatable across contexts.

**Caps (Q-sentences)**: Embeddings of figurative usages scatter outward in many directions, forming broad "caps" that occupy multi-lobed shapes, overlap arbitrarily with other words' caps, and resist systematic compression.

There is extensive literature on children acquiring language that seems to confirm the idea that P-sentences form 'the core' of how a word is used at first in its 'label' (or 'object-word') function, while Q-sentences are uses acquired later on, when idiomatic and metaphorical uses start to be employed. There is an intriguing connection here to cognitive studies but this exceeds our scope which remains philosophical at its core: are our sentence patterns in the semantic space 'real patterns'?[^3]

### Object-word and word-concept

In our model we can introduce two constructions, defined first as the centroids of the two different distributions:
- '**object-word**', as the centroid of the P sentences distribution, giving the 'core' of the distribution of all literal use sentences. This usage is primary and can be seen as a 'label' use without attached conceptual meaning; it simply describes properties of the coresponding object, like in "cat meows". 
- '**word-concept**', as a name for the entire cap formed by idiomatic, metaphoric and non-literal use sentences, like "he let the cat out of the bag". On a meta-level, both object-word and word-concept are labels for two different patterns we can see in a LLM sentence embedding space.

### Some empirical findings

Below I have attached 3 graphs that resulted from the projection of different sentences in a subspace that includes the "real axis". In the first one, a cillindrical projection is employed to analyze the distribution of P/Q sentences for different words. The P rods emerge (more clearly for some words than for others) and also the more fuzzy, distanced Q cap can be seen[^4].

![Caps and Rods](/Users/florin/Library/Mobile Documents/com~apple~CloudDocs/--ȘCOALA ESEURI/Modern Metaphysics/figures/rods_and_caps.png)

The second one shows how "cat" and "tomato", despite overlapping their Q caps, have only one common sentence in our specific set, while "cat" and "fork" overlap but have none in common. This indicates the potential for non-literal, metaphoric use of the words in Q-cap sentences. For instance, "A cat is a furred fork" seems not so completely absurd because there is actual allready potential overlap:

![Overlaping Caps](/Users/florin/Library/Mobile Documents/com~apple~CloudDocs/--ȘCOALA ESEURI/Modern Metaphysics/figures/caps.png)

The last graph, plots the distinct area where definition sentences (their different types) can be found. In a 3 D graph this is more striking, as they are far above most of the P and Q type sentences, suggesting a third "abstraction" axis in the semantic space, one is the actual "rod" that we talk about here, connecting literal, "label" word use to its most absolute abstraction (more about this below).

![Definitions](/Users/florin/Library/Mobile Documents/com~apple~CloudDocs/--ȘCOALA ESEURI/Modern Metaphysics/figures/definitions.png)

### Rods and Dennett's Real Patterns

The rod-clusters, anchored by the 'object-word', seem to satisfy Dennett's criteria for real patterns. First, they are highly compressible: once the pattern is defined by a number of sentences, they contain in a small semantic space a very large quantity of information. If one asks you "what a cat is?" you can point him to this limited area in the semantic space where he'll find all possible literal sentences with "cat". In this sense, the object-word 'cat' offers the maximum compression possible to send a staggering amount of information which all literal-use sentences with 'cat' constitutes.[^5] Second, rods have predictive utility—knowing a rod-pattern helps predict the expected behavior of the real objects they designate. Finally, rods generalize: the same compact structure reappears across different corpora and models.

Dennett certainly does not hold that spoken or written words are fixed, atomic entries. Instead, he writes:

> “The process that produces the data of folk psychology, we claim, is one in which the multidimensional complexities of the underlying processes are projected through linguistic behavior, which creates an appearance of definiteness and precision, thanks to the discreteness of words.” [@dennettRealPatterns1991, p.45]

Our model may be a good representation for this "multidimensional complexities projection". And, immediately thereafter, quoting Churchland’s formulation of the same point, he seems to anticipate what we see today in LLMs:

> “A person’s declarative utterance is a ‘one-dimensional projection—through the compound lens of Wernicke’s and Broca’s areas—onto the idiosyncratic surface of the speaker’s language—a one-dimensional projection of a four- or five-dimensional ‘solid’ that is an element in his true kinematic state.” 

These passages show that for Dennett a word functions not as a static token but as a focal “projection” or “center of gravity” within a far richer, higher-dimensional pattern of cognitive and behavioral regularities; in other words, as a _real_ patterns.



### Caps and Nelkin's Anti-Realism

Nelkin argues that we cannot recognize a belief-pattern until we already possess the relevant propositional-attitude concepts—making the concept epistemically prior to the pattern [@nelkinPatterns1994, p.62]:

> Of course, in some sense, until we have a concept of anything, X, we cannot sort instances under X. But, here, I am claiming something stronger. For instance, presumably, experiencing cats is relevant to our acquiring the concept ‘cat’. Only because we perceive token cats, patterned as cats, are we able to acquire the concept ’cat’. But the claim here is that we cannot even discern token belief-patterns (aspatterns) until after we already possess propositional-attitude concepts. If so, the existence of the patterns can hardly give rise to our propositional-attitude concepts. To claim that the concepts originate from observing the patterns would have it upside down. This sort of Neo-Behaviorist account of the attitudes would be no more successful than bodily-movement Behaviorism. Only because we already possess propositional-attitude concepts are these patterns revealed to us at all.

But in my rod/cap framework, this chicken-and-egg structure is avoided entirely. The rod is assembled empirically, as a body of appearances—sentences in which a word is used across diverse contexts. The cap emerges from the conceptual use of the word in different Q sentences (and this may correspond better to what Nelkin calls "the concept cat" above). These distributions do not change when we look from a different angle, they just _appear_ different; however, they may vanish if we choose a different set of semantic axes.

It is not that we see the pattern because we have the concept; rather, we form the concept because we encounter and respond to the pattern. Caps are not imposed by prior concepts but are stabilizations of functional use—resolvable from the rod without assuming prior interpretive categories.

### Do we carry a LLM in our brains?

The short answer is: not really. Our language is built in a much longer training period (up to 20 years) by being continuously immersed in the practice of language while experiencing reality. There may be some "tight" pattern formed by our neural network in association to literal uses of 'cat'. But what I call 'object-word' in a LLM may be a different beast from this pattern, because of our anchoring in reality and senses.

What we can see in LLM it is not what happens in our brain, but literally _patterns of use_ in language, specifically the fact that we use any word in two very different manners. A LLM is not reconstructing our mind or cognition, it simply looks for patterns in language, through vast amounts of text, close to all of what humanity produced up to today.[^6] Our model points to a fundamental dichotomy in how we _use_ words, not in how our brains work.

## A Crucial Discovery: Definitions vs. Literal Centers

Computing the centroid of the P set—the most perceptually grounded uses—gives an empirical anchor point for the object-word: a statistical center of the word's appearance across literal contexts. Interestingly, this centroid lies beneath and offset from the cluster of definitional sentences like "A cat is a mammal," suggesting that intensional definitions are not the semantic core of the word, but rather a compressed projection that stabilizes certain generalizations.

This aligns with Dennett's view that apparent precision of linguistic definition masks a much higher-dimensional pattern of cognitive regularity. The object-word is not a fixed token or entry, but a center of gravity within a distributed field of appearances—a real pattern empirically discoverable through topological regularities in language use.

## Philosophical Conclusions

One important consequence derives from finding a literal vs. idiomatic axis structuring any semantic corpus. This axis indicates a link to the ancient philosophical problem of how a word "hooks" to the real world and to the equally old problem of what "concepts" really are. My model suggests that there is a link between pre-verbal patterns of sensations with patterns of word usage in sentences, proposing a somewhat[^7] new definition for concepts.

A dual structure, with a hard core and a flexible halo acquired through time and experience, can be theorized for all sorts of patterns, including visual patterns. This may be possible because all our knowledge implies a connection from the outside world to our mind and lives always under the tension of "hard, objective truth" and "personal, subjective interpretation."

The pattern of the cap (i.e., the meaning of the word) is a pattern of sentences—a pattern of word use, acquired through use rather than definition. This approach offers sollutions to some problems about meaning in general: how can meanings evolve and still be coherent? How can we have slightly different meanings for words and still communicate? It also sheds light on Quine's indeterminacy of translation: "gavagai" is not to be translated by observing actions but by looking at the pattern of sentences the word is used in.

My model shows that Dennett's mild realism description of patterns is closer to what actual patterns inside an LLM really are. It is difficult to understand the rod and cap distributions as conceptual artifacts in Nelkin's sense because they do not depend, as patterns, on the theoretical method we use to look at them.

## Patterns as Projections

What seems to be at work always in pattern recognition is projection: we project onto the objects of the world the content of our minds when we recognize them. But in my model, the 'cap'—which is the meaning of the word in its totality—constantly grows since our infancy when we acquired language, being constantly projected and adjusted as a result of feedback from projection. If I only knew black cats all my life, "this is a white cat" would not be a sentence in my 'cat cap.' But once I see a white cat, the cap increases to accommodate the reality check.

More broadly, any pattern of meaning—even attitude-propositional or visual—can be conceived as a dual structure with a stable core that anchors shared objectivity and a fluid halo that shifts with individual experience. The patterns we see emerging are 'use-patterns' and not statistical averages, directly correlating meaning with use through real data from sentence embeddings.

---

[^1]: All along this essay we'll see how the projections of sentences in a given sub-space, creates graphical patterns. But these visual patterns encode in reality **patterns of use**, something impossible for us to quantify in the absence of a LLM. 
[^2]: At this stage is more like a point, but later on, when we switch to 3D and we discover how definition sentences project, this emerges as an elongated feature.
[^3]: Even the philosophical discussion can stray in very different directions, starting with more modern approaches to patterns (Shannon's information theory, patterns in mathematics) and ending with most of what can be defined as 'reference theory' in language philosophy. Many important ideas can be brought in but this risks to turn an already too long essay in an impossibly long one.

[^4]: In this specific image there are some artifacts, but further work eliminated most of them. This is work in progress, and more in depth testing and research is needed to fully confirm these results at a scientific standard. But for the purposes of this essay, they seem real enough.
[^5]: Language itself seems to work as a 'high compression tool' for experience patterns.
[^6]: There is also the valid question of the reality anchor in LLMs, it is clear that there is none at the time. The "object-word" is simply a reconstruction, based on patterns of language use, that a LLM does for something truly connected to senses and reality in our mind.
[^7]: The true novelty is disputable to the extend that a lot of ideas from modern Philosophy of Language points to this, starting with Wittgenstein's "meaning is use". But to my limited knowledge an identification of the totality of the idiomatic and metaphorical sentences with the "concept" is not yet proposed.
