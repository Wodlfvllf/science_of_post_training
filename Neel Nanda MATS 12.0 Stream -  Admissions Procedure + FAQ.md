## 

[How my research interests have changed](#how-my-research-interests-have-changed)

[Suggested Research Problems](#suggested-research-problems)

[Pragmatic & Applied Interpretability](#heading=h.og7ealz34abn)

[Model Biology](#model-biology)

[Understanding weird behaviour](#heading=h.25i31f64m4he)

[Reasoning Models](#reasoning-models)

[Interesting phenomena](#interesting-phenomena)

[Circuit analysis](#circuit-analysis)

[Objectively Measuring Interpretability](#objectively-measuring-interpretability)

[Science of Model Character](#science-of-model-character)

[Model Forensics](#heading=h.9chbglrlshuu)t

[Science of Post-training](#science-of-post-training)

[Alignment Training](#alignment-training)

[Science of Generalization](#science-of-generalization)

[Applied Interpretability](#applied-interpretability)

[Basic Science](#basic-science)

[Novelty](#novelty)

## How my research interests have changed {#how-my-research-interests-have-changed}

My research interests have changed substantially over the past couple of years, and many of the topics I've done past work on are no longer the topics I'm most excited to supervise \- **this is often misunderstood by applicants, so please read this section**\! “Projects that interest me” is not super well defined, so below I’ve tried to give many examples of topics and directions I feel excited about.

The two biggest shifts:

1. Within interpretability, I'm fairly pessimistic about ambitious reverse-engineering, and excited about interpretability that tries to do something useful, measured against baselines, on models and problems that matter (or are good proxies for those that do). See [A Pragmatic Vision for Interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) for more  
   1. Concretely, this means "pure" interp for its own sake no longer interests me that much: grokking, circuit finding for its own sake, SAE hill-climbing, toy models, very theoretical work. But interp that could plausibly help make AGI safer very much does\! Being able to read an AGI’s mind should be extremely useful.  
2. I’m also generally interested in a bunch of safety research at varying degrees of interpretability adjacent \- broadly things which involve needing to do good science, have empirical feedback, and I can see ways it could help reduce AGI x-risk. E.g. the science of model character, model forensics, the science of post-training, alignment training, and the science of generalization, detailed below.

Some other resources:

* My [80,000 Hours podcast interview](https://80000hours.org/podcast/episodes/neel-nanda-mechanistic-interpretability/) is a good source on my takes and why they changed (though I’ve refined them somewhat since then)  
* A talk series I gave to my MATS 9.0 scholars about:   
  * The big picture of [what matters right now in mech interp](https://www.youtube.com/watch?v=XZX_CFfVgIc)  
  * How I see [mech interp helping make AGI safe](https://www.youtube.com/watch?v=XB_7OVLxkpU)  
  * [The story of sparse autoencoder research in mech interp](https://www.youtube.com/watch?v=Tgq7E4YcPKQ) and mistakes I made here, which sparked many of my changes in perspective

If you hear all this and are like, “that sounds really boring, I am no longer interested”, then great \- we probably wouldn't have been a good match\! It's much better to learn that now than later. There's a bunch of other [MATS mentors](http://matsprogram.org) who'll be opening applications soon, hopefully one of them is more aligned with what you're looking for.

## Suggested Research Problems {#suggested-research-problems}

The below are a bunch of recommendations for things I would be excited about. Strong applications often riff off of these ideas \- coming up with their own approach, but along similar themes to the below. You should not feel constrained to the problems on this list, but hopefully it can serve as some guidance for the types of questions I'd be excited to see.

**Warning**: The ideas below have **not** been filtered for “I am confident someone could make progress on this in 20 hours”. Pick something where you have some idea of how to get started (or read around the field a bit and try to generate ideas and a sketch plan before picking a problem), and expect to need to scope the ambition down as the project goes on. Do not trust LLM time estimates, in my experience they’re super off

### Model Forensics

A particularly important application of model biology is [model forensics](https://arxiv.org/abs/2606.26071): when a model has taken a seemingly sketchy action, can we figure out the motivations, especially whether it was true misalignment or has a benign explanation like confusion. Please look at [our paper](https://arxiv.org/abs/2606.26071) if you want to do a project here, there’s a lot of relevant advice\!

* Good settings for sketchy behavior are included in our [model forensics paper](https://arxiv.org/abs/2606.26071) and [task gaming blog post](https://www.lesswrong.com/posts/HACauvWhEdC6QhdS4/why-do-models-task-game)  
  * Rich datasets that may contain interesting things are [weird chat](https://transluce.org/weirdchat?ref=bounded-regret.ghost.io) (weird behaviors automatically found in open models) and [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) (real agent transcripts of various models)  
    * It’s OK but not ideal to study a transcript on a model other than the one that made it. Only do this if resampling with the new model recreates the weird behavior  
* As discussed in our paper, the strongest techniques here seem to be reading the chain of thought to form hypotheses, and constructing precise changes to causally test counterfactuals  
  * I’m pretty interested in whether you can take a setting where the model acts plausibly deceptively and this does *not* work, and see if you can understand this better with more involved techniques (chain of thought resampling, internals based methods, etc)


* Good projects here look like taking some instance of sketchy behaviour and doing a deep dive into what’s going on and what drove it, and trying to form rigorously backed conclusion  
  * My alum Aditya Singh [suggests several good projects here](https://docs.google.com/document/d/1v6bv3JQLrfdI7Mh_kVivm-Th1omm7Dw7ddTG-WQruLc/edit?tab=t.0)  
* Can you make a good agent to automatically do forensics for you? See [auditing agents](https://alignment.anthropic.com/2025/automated-auditing/) and [petri](https://www.anthropic.com/research/petri-open-source-auditing) for an open source starting point  
* The idea of using [synthetic document finetuning](https://arxiv.org/pdf/2510.17941) to teach the model false facts, and [using these to perform better forensics](https://arxiv.org/abs/2607.18966), is fascinating to me \- how useful is this? How well does it work? What else can you do with it?  
* One major issue in forensics is eval awareness. Can you better characterize [eval awareness in the best open source models](https://www.lesswrong.com/posts/mDriHK4beN5rq2tAA/verbalized-eval-awareness-inflates-measured-safety)?   
  * How well does [steering against it](https://arxiv.org/abs/2510.20487) work for mitigating it? What are the side effects? Should we trust this in a real eval?  
  * How well can we probe for it in real models? If we ran this over evals where only 1% showed eval awareness, would we just get tons of false positives?  
* I’m also interested in deep dives into examples of weird but not necessarily misaligned behavior and trying to really understand what’s going on, and ideally fix it? Example: debugging [why it thinks 9.8 \> 9.11](https://transluce.org/observability-interface)

### Model Biology {#model-biology}

This is about studying the high-level, qualitative properties of a model's behavior, treating it like a biological organism we're trying to understand. This is especially interesting for weird, emergent, or safety-relevant phenomena.

#### Reasoning Models {#reasoning-models}

* **Chain of thought faithfulness**: An obvious safety strategy is “let’s just read the chain of thought and see if it thinks anything suspicious”. Can you find evidence for or against this being effective?  
  * Can you learn anything by studying examples of unfaithful chain of thought? Eg from [Arcuschin et al](https://arxiv.org/abs/2503.08679) or [Chen et al](https://assets.anthropic.com/m/71876fabef0f0ed4/original/reasoning_models_paper.pdf)  
  * Can you create situations where the model does use suspicious reasoning? E.g. by training the CoT to be distorted, or telling it that its chain of thought is being monitored \- does it do the bad behaviour but without suspicious reasoning? How complex can the behaviour be?  
    * I would love to see someone use [synthetic document fine-tuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) to train a model to believe that it should have unfaithful chain of thought and see if we can interpret what's happening.  
  * Can you design good monitors or metrics for whether the chain of thought is telling us what we think?  
  * Can we tell when a CoT was causally important for a model giving its answer?  
    * This is a subtle concept to pin down – models may have a causally irrelevant CoT that gives the same answer. And [editing the CoT throws the model off distribution, which may change its answer for unrelated reasons](https://www.alignmentforum.org/posts/HQyWGE2BummDCc2Cx/the-case-for-cot-unfaithfulness-is-overstated)  
  * What factors lead to different forms of “unfaithful” CoT? Can we distinguish them?  
    * Deliberately rationalizing a preconceived answer ([Arcuschin et al.](https://arxiv.org/abs/2503.08679), [Lindsey et al.](https://transformer-circuits.pub/2025/attribution-graphs/biology.html))  
    * Models changing their answer because of a hint but not admitting it ([Chen et al.](https://assets.anthropic.com/m/71876fabef0f0ed4/original/reasoning_models_paper.pdf))  
    * Models taking logical shortcuts in maths problems after getting stuck, to claim they’ve achieved a valid “proof” ([Arcuschin et al.](https://arxiv.org/abs/2503.08679))  
    * Models giving a reasonable chain of thought, but at the last minute “flipping” to a different final answer ([Arcuschin et al.](https://arxiv.org/abs/2503.08679))  
* **Thought anchors**: In [Bogdan et al](http://thought-anchors.com), my scholars present a paradigm for what mech interp could look like for reasoning models, where we study sentences as our main unit of analysis, and use tools like resampling to understand which sentences are important, and do causal interventions to understand the dependence between pairs of sentences. How can you extend and build on these techniques? Can you find anything interesting by using them? Can you find any weaknesses or limitations?  
* **Steganography**: Can models encode information in their chain of thought? Ideally, in a way where the chain of thought is still plausible to us, but even in ways where we know it's encoded but we don't know what it means. Can you train a model to have reasoning we don't understand and use interpretability tools to decipher it? [Relevant work](https://www.alignmentforum.org/posts/ywzLszRuGRDpabjCk/do-reasoning-models-use-their-scratchpad-like-we-do-evidence)  
  * Note that this needs to allow the model to do tasks it couldn't do without a chain of thought to be interesting. 

#### Interesting phenomena {#interesting-phenomena}

* **User models**: [Chen et al](https://arxiv.org/abs/2406.07882) shows that LLMs form surprisingly accurate and detailed models of the user, eg their gender, age, socioeconomic status, and level of education, and do this from very little information. They can find these with probes, and steer with these to change the model’s actions in weird ways.  
  * This is wild\! What else can we learn here? What else do models represent about the user? How are these inferred? How else do they shape behaviour?  
  * Do LLMs form dynamic models of users for attributes that vary across turns, eg emotion, what the user knows, etc.  
    * As a stretch goal, do LLMs ever try to intentionally manipulate these? Eg detect when a user is sad and try to make them happy  
* **Out Of Context Reasoning:** Sometimes models generalize much further than expected. Most famously, [emergent misalignment](http://emergent-misalignment.com), where training a model to write insecure code turns it into a Nazi. What's up with this? Some past work from my scholars suggests this is often downstream of learning a [single](https://arxiv.org/abs/2507.08218) [direction](https://arxiv.org/abs/2506.11618), with hints that it's because the general solution is [more efficient](https://www.alignmentforum.org/posts/gLDSqQm8pwNiq7qst/narrow-misalignment-is-hard-emergent-misalignment-is-easy). But there's a lot we don't understand \- is this the whole story? Why are some solutions easier to learn than others? Do these weird effects come up in any real use cases?  
  * A notable example is [synthetic document fine-tuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/), where training on LLM-generated documents from a world where some false fact is true can get LLMs to internalize it and act on the consequences of that false belief. What’s going on here? Does this really work? How robust is it? Etc.  
* **Concept Representations**: How are specific interesting concepts computed and represented?  
  * Can we train a [truth probe](https://arxiv.org/abs/2310.06824) that generalizes well to real situations?   
  * What about a [deception](https://arxiv.org/abs/2502.03407) probe?  
  * How is [uncertainty](https://arxiv.org/abs/2406.16254) represented?   
  * Why on earth is there a [misalignment](https://arxiv.org/abs/2506.11618) [direction](https://openai.com/index/emergent-misalignment/)?  
  * How is the [awareness](https://www.alignmentforum.org/posts/E3daBewppAiECN3Ao/claude-sonnet-3-7-often-knows-when-it-s-in-alignment) of whether or not it is being [evaluated](https://arxiv.org/abs/2507.01786) [represented](https://arxiv.org/abs/2505.14617v2)? Nemotron 49B seems like a good model to study here.  
* **Conflicting information**: How do models deal with conflicts between instructions or goals, or their prior knowledge and the context?

#### Circuit analysis {#circuit-analysis}

* **Attribution graphs**: Are [attribution](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) [graphs](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) a pragmatically useful technique for understanding model biology? Try playing with the graphs on [Neuronpedia](https://www.neuronpedia.org/gemma-2-2b/graph). Can you find things with them that cannot be found with simpler techniques like guessing and checking?  
  * How important is precision? One notable consequence of the attribution graph approach vs, e.g. prompting, is that it can find much more nuanced and detailed hypotheses, like the addition analysis in [Lindsey et al](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-addition). Are there tasks where this precision is important?  
* **Baselines**: There's a bunch of simple methods that fundamentally boil down to guessing hypotheses and checking them. Far more effort has gone into fancy techniques like attribution graphs than these. How far can we push them?  
  * Linear probes can be highly effective at identifying concepts the model is representing – can we automate and scale the process of testing many linear probes, at all appropriate layers / token positions, for a given task?  
  * Scaling the process of reading a model’s chain of thought. How can we best analyze and aggregate them to look for unexpected properties, across many prompts? [Docent](https://transluce.org/introducing-docent) is one interesting approach in this direction.  
  * Simply observing model behavior in response to an appropriate mix of prompts can be highly effective to infer mechanistic hypotheses, but there’s an art to doing it well. What do best practices here look like? Can they be automated?  
* **Automation**: Can we automate the full hypothesis generation \+ validation loop with [LLM agents](https://alignment.anthropic.com/2025/automated-auditing/)?  
  * Automated hypothesis generation  
    * Can LLMs simply guess the high-level casual graph of a task? Can an agent make more headway if we let it iteratively choose diverse prompts and read the output  
    * How good are LLMs at interpreting an attribution graph and how good can we make them with the right prompt and scaffold?  
  * Automated validation  
    * Can we automate the design of probes to test for the presence of predicted features?  
    * Can we automate intervention experiments, and synthetic / out-of-distribution inputs, used for hypothesis validation?

#### Objectively Measuring Interpretability {#objectively-measuring-interpretability}

* **Eliciting Latent Knowledge:** Can we use interpretability to elicit secret knowledge from a model? What techniques work best?  
  * In [Cywiński et al](https://arxiv.org/abs/2505.14352) my scholars taught a model a secret word by training it on descriptions of that word, and then retrieved it with both black and white box techniques. Can you do better? ([their models](https://huggingface.co/collections/bcywinski/gemma-2-9b-it-taboo-6826efbb186dfce0616dd174))   
* **Understanding-based downstream tasks**: In addition to the above, what other objective tasks are there that test our success at understanding? [Movva et al](https://arxiv.org/abs/2502.04382). is another nice example.

#### Model Diffing

Model diffing: What changed when a model was fine-tuned? 

- Black box [diffing agents](https://www.alignmentforum.org/posts/qi4mNbZYAFDYwfRba/building-and-evaluating-model-diffing-agents) work surprisingly well, I’d recommend starting here. I really liked the idea of introspection adapters. Lots of room to do better, and to find real use cases.  
- I’m particularly excited about the use case of using model diffing to help identify what changed in alignment training, bugs in datasets, and help improve things, e.g. can you improve [character training](https://arxiv.org/abs/2511.01689) or [model spec midtraining](https://arxiv.org/abs/2605.02087)?  
- [Narrow finetuning leaves readable traces](https://arxiv.org/abs/2510.13900) was a fascinating result to me \- why does it happen? Is that diff vector just a bias term representing “you are on the topic of the fine-tuning domain” or something deeper?

### Science of Model Character {#science-of-model-character}

Models have something like a character: values, personas, self-models \- often not the ones we trained for, and we barely understand how they work.

- Deep dives into character phenomena: Take a striking result from the literature and figure out why it happens (e.g. basically any Owain Evans paper).   
  - E.g. [Value Leakage](https://arxiv.org/abs/2607.14345) shows a model's answers are silently shaped by its own values \- why? Where do the values intervene? Can you make it disclose, or turn the effect off? Can you combine it with [Gilg et al](https://arxiv.org/abs/2605.13339) to find a linear direction predicting it and causally mediating it?  
- What are [a model's value rankings](https://www.alignmentforum.org/posts/k6HKzwqCY4wKncRkM/brief-explorations-in-llm-value-rankings), and do they predict behaviour? Where do they come from?  
  - Do models follow their stated principles? E.g. my scholars [red-teamed whether constitution-trained models follow their constitutions](https://arxiv.org/abs/2605.24229)  
- Where does [the assistant axis](https://arxiv.org/abs/2601.10387) come from? What does it actually do? Can interpreting it in more detail, e.g. with SAEs or J-Lens, tell us anything meaningful about what post-training does? If so, how does this differ across models?

### Improved Interpretability Methods

I am excited about having generally useful methods for understanding what’s going on in a model during its forward pass, typically by interpreting activations. Probes and sparse autoencoders are classic ones here, I’m excited about newer ones like J-Lens and natural language autoencoders (and the general idea of [meta-models](https://www.youtube.com/watch?v=Aroazwb_QW8) though I’m moderately more cynical than I was when recording that video)

* I’m particularly interested in improvements and red-teaming of new and promising ones. This can look like using them as a tool in a complex and realistic use case and seeing how well it works ([example on activation oracles](https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use)), or probing into potential flaws and designing evals for these, or trying to improve on these flaws and making good evals to show this ([example on activation oracles](https://arxiv.org/abs/2606.02609))  
* A rich source of real world data to play with is [SWE-Chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) a bunch of agent transcripts  
* This includes meta-models / interpretability foundation models, I’m especially curious about natural language autoencoders  
* If you want to argue that a method is useful, remember to compare to baselines\!

* [J-Lens](https://transformer-circuits.pub/2026/workspace/index.html) attempts to find the intermediate variables in a model’s forward pass ([my review](https://www.alignmentforum.org/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper), [Neuronpedia demo](https://www.neuronpedia.org/jlens)). Based on the paper it seems to work reasonably well\! What can you do with it?  
  * **Key resource**: [Open source J Lenses](https://huggingface.co/camilablank/workspace-lenses/tree/main) from my scholars Camila and Agam on a bunch of models from Qwen 3.5 4B to deepseek v4 flash  
  * Being single token is a crippling limitation. How well do the multi-token J Lenses in the appendix like template lens and oracle lens work? (see [open source template lens](https://huggingface.co/camilablank/workspace-lenses/tree/main/qwen3.6-27b/template-lens) and data from Agam and Camila)  
  * From a scientific perspective, what is J-Lens actually doing? How much better is it really than logit lens and tuned lens and why? How much does it hallucinate?  
* [Natural language autoencoders](https://transformer-circuits.pub/2026/nla/index.html) try to autoencode activations as natural language and back, [Neuronpedia demo \+ open source](https://www.neuronpedia.org/nla). What can you do with them? Do they actually work for tasks of interest?  
  * **Key resource**: This [Qwen 3.6 27B NLA](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl) from my scholar Celeste, it’s a good model and should be a good quality NLA  
  * I’m particularly interested in using the activation reconstructor to measure the quality of a description, e.g. figuring out which claims can be removed and improve reconstruction accuracy to help reduce hallucinations, as briefly explored [here](https://transformer-circuits.pub/2026/nla/index.html#characterizing-nla-confabulations)

### Science of Post-training {#science-of-post-training}

Post-training shapes everything about how models behave, and we understand it poorly. How does it work, and how could we control it?

- Distillation and inheritance: Models inherit a surprising amount from their teachers \- including "hereditary diseases" you can't easily filter out. See [Why Do Naive SFT Filters For Safety Properties Fail?](https://www.alignmentforum.org/posts/wyZRNgpeiPeRXB6eT/why-do-naive-sft-filters-for-safety-properties-fail) and [Data filtering works a lot worse than you would expect](https://www.alignmentforum.org/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-a-lot-worse-than-you-would-expect). Why? Can you build a clean model organism of filtering failing, and find something that works?  
- What does each stage do? [Most of Gemini's safety behaviour comes from SFT, not RL](https://www.alignmentforum.org/posts/nLrrYweeFxgXACSmS/sft-drives-gemini-s-safety-properties-1)\! What else about the pretraining/SFT/RL division of labour is not what we assume? Olmo 3 think is a good model to study here  
- How can we steer what models learn in training (with interpretability or otherwise), i.e. [intentional design](https://www.goodfire.com/blog/intentional-design):  
  - e.g. [concept-ablation fine-tuning to steer OOD generalization](https://arxiv.org/abs/2507.16795)  
  - Or [benchmarking interventions against reward hacking during RL](https://www.alignmentforum.org/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against).

### Alignment Training {#alignment-training}

Can we actually make models deeply aligned \- aligned in ways that generalize far beyond the training distribution \- rather than just behaviourally compliant where we trained them?

- Anthropic's [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) shows that teaching the principles behind aligned behaviour (constitutional documents, stories, difficult-advice conversations) beats training on demonstrations alone. Can you extend this \- e.g. invent an SFT method (beyond difficult-advice conversations) that makes a model substantially more aligned out of distribution?  
  - [Model Spec Midtraining](https://alignment.anthropic.com/2026/msm/) is a good source of open source settings to study  
  - The key metric of success is improvement on domains far from where you trained. I encourage using more than one eval\! See e.g. [our post](https://www.alignmentforum.org/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)  
- Better evals for deep alignment: We badly need measures beyond the blackmail demo. What would a good eval of "is this model aligned in a deep way that generalizes OOD" look like? Also relevant: [red-teaming constitution adherence](https://arxiv.org/abs/2605.24229)

### Science of Generalization {#science-of-generalization}

Sometimes models generalize much further than expected, in ways that really matter for safety. Why?

- Emergent misalignment: training on insecure code turns models into Nazis. We now know [the general "misaligned persona" solution is easier to learn than narrow ones](https://arxiv.org/abs/2602.07852) and [it's often a single direction](https://arxiv.org/abs/2506.11618) \- but why is general easier than narrow? Is that the whole story? Does this show up in real training runs?  
  - The broader question here is why do models generalize to one solution over another when both perform well on training data \- emergent misalignment is just a very clean example of this being weird

### Applied Interpretability {#applied-interpretability}

I'm excited about a work that finds practical, real-world applications of interpretability, especially for safety. This isn't just using downstream tasks for grounding. The point is to choose a problem that actually matters and show that interpretability helps. I find this an exciting line of work because if we want interpretability to eventually be useful for making AGI safe, figuring out how to do things now seems like important practice.

* **Monitoring:** An extremely important problem in safety is that of monitoring: as a model runs, seeing whether a certain concept is present. The classic technique of probing is extremely cheap and is SOTA for cheap monitoring on frontier models for detecting misuse. What else can we do with probes?  
  * How can probes be improved? Can we address cases where traditional probes work less well, like when information is spread across tokens or when there is a long context with lots of room for false positives?   
  * [Kramar et al](https://arxiv.org/abs/2601.11516) from my GDM team is a good place to start  
* **Prompt injections**: Prompt injections are a big deal and no one knows how to fix them. I liked [the model given in this post](https://www.lesswrong.com/posts/d8xDGzCEYE639qqEv/a-theory-of-prompt-injection-and-why-you-should-study-roles), can you use this to construct interventions on a model that make it robust to prompt injections? (e.g. adding a constant vector depending on what turn/context the model is in)  
* **Other techniques:** Some other techniques that I think may have promising practical applications.  
  * [Conditional steering](https://arxiv.org/abs/2409.05907): applying a steering vector only if a probe fires. This lowers the side effects of steering a lot.  
  * [Training data attribution](https://arxiv.org/abs/2205.11482): A family of methods, including influence functions, to study which data points would have influenced a model to take a particular behavior more. The mathematical claims here are basically bullshit, but I think that being able to associate model behaviors with data points opens interesting use cases like debugging or [removing noisy data points](https://arxiv.org/abs/2002.08484) or [filtering for the best data to finetune on](https://arxiv.org/abs/2402.04333)  
    * Warning: If you haven’t played with TDA before, this may not be practical to work with in 20 hours  
  * [Abliteration](https://arxiv.org/abs/2406.11717): In refusal is mediated by a single direction my scholars cheaply jailbroke models by removing the refusal direction from the weights. How else can the idea of “[abliteration](https://huggingface.co/blog/mlabonne/abliteration)” be applied?

### Basic Science {#basic-science}

I am generally excited about work that moves forward our understanding of key problems in interpretability. This is less of a focus of mine than it used to be, but I am still excited to supervise such work. However, I frequently get basic science projects on problems I don’t think matter / that go super into the weeds, e.g. work on toy models, algorithmic tasks, or interpretability during training. I’m excited about topics like the below

* **Understanding Reasoning Models:** What is actually happening inside reasoning models that produce long chains of thought? Can we [intervene](https://arxiv.org/abs/2506.18167) on their reasoning process?  
  * It's surprisingly difficult to edit a model's chain of thoughts, since if you regenerate from that point onwards they will often immediately correct any errors introduced. What's up with this? Can we stop it? If you token force the next sentence, is that enough? Etc.  
  * How do models trained with RL compare to those that are distilled from an RL-trained model? E.g., comparing QwQ to an R1 distill.  
* **Steering Fine-tuning**/**intentional design**: In [Casademunt et al](https://arxiv.org/abs/2507.16795) my scholars showed that you can control how a model generalises after fine-tuning, with zero change to the data or loss, by ablating concepts we don't want it to use. They used this to mostly fix [emergent misalignment](http://emergent-misalignment.com). This is really cool\! Where else can we apply it?   
* **Why do filler tokens work**: A wild fact about modern LLMs is that adding a bunch of meaningless dots between a maths question and the answer (with no CoT) improve performance. Why?\! What algorithm is being performed? Is it truly taking advantage of the parallelism? [This paper](https://arxiv.org/pdf/2607.03502) is a good place to start. Deepseek v4 flash is 300B total params and benefits from filler tokens, and has [J-Lens available](https://huggingface.co/camilablank/workspace-lenses/tree/main)

### Novelty {#novelty}

* **New ideas**: For anyone feeling ambitious, I’m extremely impressed with any application showing ideas and applications of interpretability that are new to me or that I didn’t expect to work   
  * One of my favorite recent examples was in [Casademunt et al](https://arxiv.org/abs/2507.16795), where my scholars showed it was possible to steer finetuning without changing the data.





by the time I return you can do few stuffs for me. See we can build tools like whatever project we choose, we need tools to inspect , investigate probes etc. So to do that efficiently we need to build a comprehensive toolkit, modular clean and efficient and that could be used for running through experiments fast and correct. First brainstorm the ideas find what tools experiments we need to to run the code. And build a plan to build the interpretability research.