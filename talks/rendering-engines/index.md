---
title: "Rendering Engines of a Textual World"
subtitle: "What LLMs Reconstruct"
date: 2026-05-10
venue: "Beyond the Imitation Game, Bucharest"
description: "An LLM works like a virtual-reality rendering engine: a stored geometry with no camera of its own, rendering only when a prompt supplies the viewpoint."
categories: [LLMs, PRU]
---

The talk makes one claim: a large language model works more like a virtual-reality rendering engine than any other software artefact we know. A VR engine stores a view-independent scene — geometry, materials, lights, relations — in a multidimensional space as matrices and computes a 2D image from it only when a camera pose is supplied; same model, many renders. An LLM builds a high-dimensional geometry out of one-dimensional token streams in training, like a model of the "text world". When prompted, it then returns a one-dimensional stream; the prompt does what the camera does, selecting the region of that geometry the output will traverse. The argument is that this is the same operation, not a picture of it. Both systems carry a stored structure through intermediate coordinate systems to a channel-constrained output — model space to camera space to screen in one case, token embeddings to contextualized hidden states to next-token scores in the other. A selector activates a region; a transformation compresses that region into the output channel. "Next-token prediction" is an accurate description of the channel and silent about the geometry the stream is a compression of. Prediction is what rendering looks like when the channel is serial.

The selection is checkable. Sampling a hundred completions of a single prompt from a local model at high temperature and embedding each as a sentence vector, the points do not scatter through the space: they lie on a low-dimensional structured surface inside it. The prompt fixes a region, not an answer. The implications follow from the engine having no camera of its own. The geometry is fully present and fully perspectiveless; nothing renders until a viewpoint is supplied from outside. The model therefore has no default standpoint — no ego, a renderer without opinions — and it can render mutually contradictory views from the same store, since the geometry holds every position its training data licensed. Two further consequences close the talk: LLMs are instruments for studying language and mind, much as the telescope and the microscope were for Early Modern science; and our own answering may work the same way — we bring forward "the manifold" our answer lies on before we find the words on them.

[Slides (PDF)](slides.pdf)
