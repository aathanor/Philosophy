1. **What exactly is realism (before Dummett)?**
    1. Problem of Universals: realism/nominalism
    2. Epistemological view: realism/idealism
    3. Scientific view: realism/instrumentalism
    4. Mathematics: realism/platonism

   There is a shift, from metaphysics to semantics (or more from **truth-conditions understood metaphysically** to truth-conditions understood **epistemically** or **linguistically**): what is real independently of our conscience observing it, and what sentences are true independently of our mind assessing them.

2. **How is Dummett framing the problem?**
    1. He is approaching the problem from Philosophy of Language: what is real is described in propositions about the world.
    2. Example: **"Somewhere in the woods, a mushroom is growing."**
        1. Classical (epistemological) realism:
            1. **Realism**: "Yes, the mushroom is growing regardless of our knowledge." The world behaves independently of observers.
            2. **Idealism**: "If no one can observe the mushroom, its growth lacks meaningful description."
        2. In Dummett's framing, **semantic realism** holds that the proposition has a truth-value even if verification is impossible. **Semantic anti-realism** denies this:
            1. **Realist**: The sentence is meaningful because it has a truth-value, whether or not it is known.
            2. **Anti-realist**: If it cannot in principle be verified, it lacks determinate truth-value and is semantically void.

3. **Sticky point in understanding Dummett's view:**
    1. Critics say realism is about existence, not truth. (e.g., Devitt)
    2. But existence claims require **language**, and truth-values are inseparable from meaning when existence is claimed linguistically.
    3. Dummett’s concern is not metaphysical existence, but **how we attach meaning to existence claims** through language.

4. **Dummett's definition of Realism:**
   Realism is about the **interpretation of a class of statements** (not about the world). The debate is about whether these statements have truth-values **independently** of our means of verification.

   > "A dispute over realism... relates to a class of statements... which I shall term the *disputed class*."

   > "The dispute... concerns the notion of truth appropriate for statements of the disputed class; and this means that it is a dispute concerning the kind of meaning which these statements have."

   - Realism implies **bivalence** and the **excluded middle**: each statement is determinately true or false.
   - Thus, classical logic applies to the disputed class.
   - Anti-realism rejects this: meaning must involve **verifiability**.

   
5. **The Role of Quantifiers (illustrated with the mushroom):**
    - The sentence "Somewhere in the woods, a mushroom is growing" becomes:
      > **∃x (InWoods(x) ∧ Mushroom(x) ∧ Growing(x))**
    - Classical logic assumes the domain (woods) is given and complete; truth is defined model-theoretically.
    - But for the anti-realist, **verification** is essential. If we can’t find such an x, the sentence lacks a truth-value.
    - The problem is generalized: **quantifying over open or infinite domains** makes verification impossible.
    - Thus, **bivalence fails** in such domains.

6. **The clash with Frege and Tarski:**
    - Frege assumes all well-formed thoughts are true or false—he presupposes realism.
    - Tarski defines truth model-theoretically, without addressing **epistemic access** to domains.
    - For both, **truth precedes understanding**.
    - Dummett reverses this: **meaning precedes truth**. If you don’t know what counts as verifying the mushroom’s existence, the sentence’s meaning is unclear.

7. **Semantic theory, semantic value, and meaning:**
   - **Semantic Theory**: A compositional model for determining a sentence’s truth-value.
   - **Meaning**: What the speaker must grasp to use the sentence.
   - **Semantic Value**: The entity (e.g., reference, extension, verification condition) associated to a linguistic expression by a semantic theory.
   - **Truth**: The value the sentence receives within the model (or by verification).

   Key points:
   - Semantic theory determines **truth**, not meaning.
   - Semantic value is what allows a sentence’s truth to be **compositionally explained**.
   - In intuitionistic logic, semantic values are **proof-based**, not model-based.
   - Meaning **determines** semantic value, and semantic value **determines** truth.

8. **Semantic values are system-specific:**
   - Different logics assign different types of semantic values.

| Expression Type       | Classical Logic                          | Intuitionistic Logic                        | Modal Logic                                   |
|-----------------------|-------------------------------------------|---------------------------------------------|-----------------------------------------------|
| Constants / Names     | Individual in domain D                    | Individual in domain D                      | Individual in domain D                        |
| Variables             | Assignment: x ↦ d ∈ D                    | Monotonic assignment                        | Assignment per world: w ↦ d ∈ D               |
| 1-place Predicates    | Subsets of D: P ⊆ D                      | Verifiable/decidable sets                   | Function from world to set: w ↦ P(w) ⊆ D      |
| n-place Predicates    | Relations over Dⁿ                        | Monotonic relations over Dⁿ                 | w ↦ R(w) ⊆ Dⁿ                                  |
| Function Symbols      | f: Dⁿ → D                                 | Constructive functions                      | w ↦ f(w): Dⁿ → D                               |
| Atomic Sentences      | True or false                             | Provable or not                             | True/false at world w                         |
| Complex Sentences     | Built via truth-functions                 | Via inference rules                         | Composed via accessible-world structure       |
| Quantifiers           | ∀x φ true if φ true for all x ∈ D       | ∀x φ provable for arbitrary x              | φ holds at all accessible worlds              |
| Modal Operators       | Not used                                  | Not standard                                 | □φ: true at w if φ true at all w'             |
| Sentence Value        | True or false                             | Provable or not                              | True at w or not                              |

   - Semantic values are **not interchangeable** between logics.

9. **Meaning and truth-conditions:**
   - Dummett argues (via Wittgenstein-like examples) that **understanding does not consist in knowing truth-conditions**.
   - Classical semantics fails for sentences like the mushroom sentence when no verification is possible.
   - A use-based, verificationist model is needed instead.

10. **Reductionism and its failure (illustrated via the mushroom):**
   - Suppose someone says: "To say a mushroom is growing somewhere means that I could have an experience of it."
   - But: describing that experience **requires** using the vocabulary of the original sentence ("mushroom", "growing").
   - Therefore, the **reductive class presupposes** the given class.
   - No real translation is possible.
   - Hence, **reductionism fails** to explain meaning; it may preserve truth-values, but **abandons reference**.

    Types of anti-realism:
    - **Mild anti-realism**: retains truth-conditions and bivalence, but discards reference (e.g., Frege’s directions).
    - **Radical anti-realism**: rejects truth-conditions and bivalence (e.g., intuitionistic logic, inner sensation ascriptions).

    In both cases, **meaning is no longer grounded in reference to independently existing entities**, but in what can be known or used.

    Dummett’s guiding thesis: **Truth is only meaningful where understanding is possible.**

