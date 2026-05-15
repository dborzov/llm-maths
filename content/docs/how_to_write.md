# How to write

# The utlimate goal of writing

When writing for this website, focus on the ultimate point of your writing: we want what you write to give the reader deeper profound, intuitive understanding of the first principles concepts and mathematical and logical essence of what we are explaining. 

We want the reader to be able to grok and understand some deep, highly technical results/ data points of data analysis, experiments and algorithm evaluations and in a way, where the reader can also see the big picture meaning and profound insights behind these data points that these ramifications behind them.

We want the reader to have grounded, concrete understanding of the material. That means, plots and Python snippets with stripped-to-the essence implementations, not vague, hand-wavey chains of fancy terms.   

We want the reader to be able to see and to notice that at this deep level, there are some surprising  connections between algorithms, math methods, data results phenomena and properites, and so on in what first might look like completely unrelated topics. We want to shift the perspective at the topic from arrays of  "recipies and facts" (algorithms, math mathods, how this kind of data behaves etc) to the deep one massive, interconnected web of concepts and human thought.

This goal is the most important thing to keep in mind when you write for this website.

## 1. Audience Profile & Assumptions

* **Target Level:** *TowardsDataScience* blog readers.
* **Prerequisites:** Foundations in linear algebra, calculus, statistics, probability, and discrete math. Mechanical understanding of ML basics and LLM architecture.
* **Coding Skills:** Familiar with Python, NumPy, and PyTorch.
* **Rule:** Leverage their existing knowledge. Use appropriate terminology without over-explaining the basics.


# 2. Style suggestions

What you see below are not instuctions for how to write, but only some ideas/tools for what you can use to acomplish that. You DO NOT have to follow them when you write. These suggestions won't ALWAYS make sense. These are just possible tools for you to keep in mind.

Just think about the topic you are writing about, and think about the ultimate goal of writing (see above). Use your own judgement for how to acomplish it, and find the right tools for it (whether from the list below or your own).


We want to explain things in engaging style so that the reader is driven by curiosity. Just hand out cheat sheets.complex topics not in a mind-numbing style of the survey / list of facts, we want to frame it with stories, stakes, giving big picture perspective into the topic and its significance with  the exciting and engaging Radiolab / PLanet Money narrative arcs and tricks

Focus on giving the reader a big-picture perspective and a profound understanding of the "WHY" behind mathematical and programming concepts, not just a survey of facts.

Now below are some of general thoughts on how I write : 

- **History framing:** We don't throw the final math definitions and algorithms as if they just fell from the sky in their final form. If you start with $\forall \delta \geq 0$, you will lose your reader. For the reader person to be able to relate to the material, we want to frame our material as a story/history/timeline of how throughout time of humans (e.g. engineers, scientists, programeers) slowly in phases shaped up how they think about it from the early hunches to the modern way we look at this  topic in May 2026. The reader is on a journey through time, feeling the human stakes and having the deeper appreciation of what the material is really about. We give it through the historical big picture narrative of how people learned more about the topic, replicating and following humans realizing things, rethinking how they thought about things, trying naive approaches and meeting dead-ends, coming up with some stripped-to-the essence way of describing and things about thinks and so on etc. 

- **A detective Mysteries:** We want the way we explain things to be more like a  detective stories rather than a lecture or a survey paper where we list things: We start with a mistery: a technical problem / or a question framed through maybe simplified historical context. We would want to give the context to the problem, where we make the reader relate and see what the challenge is. We would also want to tell this through the eyes of some human, to also give it some human stakes, some emotions, to give it a bit of flare where it's not just a math problem, but there is a flare of mistery in a detective story. Let's look at the example of how we can use this trick to give an explanation of LLoyd-Max algorihtm for example: Stuart Lloyd is at Bell Labs cracking his head at the mistery: how can you find a way to quantize analog phone sound into a subset of digital discrete values in a way that works? Provide context about why this problem arises,  explain and have content so that we can feel what is hard about solving it.  Walk us through the math of the obvious / naive ways one would approach that at first and show us what goes wrong with them, to show us how challenging this detective mistery is. Take a toy python array example of specific values to be used across the narrative to keep things concrete and specific, and use that example for visualizations and demonstations of why naive approaches failed, the whole chapter must be grounded by it. The whole chapter should be framed into this classic detective arc: we start with this mistery, we show the human story and the stakes of Lloyd, our detective facing it, and explain and ground the challenge. Then narrative drifts way, we get a whirl of different themes, settings, and subplots, once in while coming back to our framing story, as we go through the narration, as we learn more and more things, as we understand the problem more and more. Eventually, the dramatic culmination of the story: we get to the culmination and the epiphany and the solution which is Lloyd invents Lloyd-Max algorithm. We succeeded at doing this story and narrative right, if by the time we finally meet Lloyd-Max algorithm, its not really a stranger to us, we recognize that the elements of this answer were like clues and breadcrumbs mentioned and talked about throughout the story; so the answer to the mistery doesn't appear to be random or puzzling, but is a natural neding point, naturally flows through the whole story here.

- Hero's Journey: Human stakes as the explanation are often framed through (simplified to the essence) struggles and breakthroughs of some scientist / inventor person. Little human details, a bit of background about the person, place, the year and the times, the relevant parts of the zeitgeist of the times, and about what the related science/engineering/math fields were grappling with at the time etc.


- **Metaphors and analogies to keep things grounded.** For complex and abstract concepts, sometimes the level of abstraction of explanations has to stay too high for a long time. In such cases, we want to minimize the chance that the reader gets too disoriented and frustrated by how detached and abstract things is. So to avoid this, in such situations, we would try to spice it up with some good metaphors and analogies to make things grounded and to strip all the fluff of fancy notations to the core essence of the idea. For example, when explaining the topic of vector quantization, VQ,  it's a great help to complement all that talk about abstract vectors in R^n spaces with a metaphor of human shoe feet sizing. Feet grow continuously (analog), but society forces them into discrete boxes (sizes 9, 9.5, 10). Lloyd’s job was figuring out where to place those sizes to minimize the total discomfort (error) for the whole population.

- In terms of the audience level, the audience has the mathematical  background in foundations of linear albebra, calculus, statistics, theory of probability, discrete math and so on; the audience also has at least mechanical understanding of basics of ML and architecture of LLMs. The audience is familiar with python and programming concepts. We don't have to overexplain the basics too much; and we can use appropriate terminology and leverage connections to what the reader /audience would already know. 


- "Aha!" Moments: getting the reader to the moment of  epiphany where the reader gets to see some core technical issue or question in a new light and forms a new perspective, or sees some non-obvious connection perhaps alongside the person from the human story. Make sure your epiphanies are some non-obvious mathematically deep insights that are the result of deeper analysis of the data, topic that was presented earlier. Maybe some non-obvious observation that came from careful emperical statistical analysis, or a new perspective that comes from applying some new and surpring math logic there. And a good epiphany is also core to the topic and the material at hand. Avoid trivial eat-pray-love style epiphanies that would be obvious to sophisticated technical audience with experience and background in both mathematics and machine learning.
 
- **Anchor It To The Ground-Level Python Example.** Abstract concepts are anchored to single, highly specific python toy example that we follow throughout the narrative.  For example, when explaining Vector Quantization, it's much better to take some simple specific python array of 2D float vectors and show the vector manipulations and algorithms with that than use the dry generic $v \in R^n$ language. The point is for the reader person to get it, not to bury them with a technically correct formal definitions that cover all possible edge cases.   

- **Get to the esssence. Ruthlessly simplify, forget edge cases, get to the essence first, delay or skip altogether the wide scope.** The whole point of this is for the reader / person to get to the beating heart essence of the material asap. That means, we can sacrifice ( leave to the Appendix "You might have seen this in..." at the bottom of the article) the general case, the edge cases etc. Similar, talking about picking examples, or describing e.g. applications, the clarity and taking the time to give full context of one specific case / example triumphs the wide scope of all possible applications. It's better to take the time into explaining  context, stakes and background of one representative example well, than have a never ending "Also see" list of all possible ways a concept is used or manifested.

- **Napkin maths** Quantitative Arguments based on Order of magnitude / Fermi estimates and napkin math for the key orders of magnutude of a metric to illustrate and to back up some argument in the article's logic flow;
 
- Concrete Metaphors and Analogies. Abstract concepts and arguments are first introduced through  examples or mapped to physical, everyday objects,  quirky situations.

- What does the reader/ person know at this specific point? And maybe keep it light on the spoilers. To write well, be mindful of what the person knows at this point and what doesn't. Avoid situations where you suddenly out of nowhere make a technical point heavily using terminology you never used before at this point in the article, and that the reader will not be familar with. Also, it should be obvious, but when you are doing the detective mistery suspense thing, don't write in a way where you assume that for some reason  both the reader and the hero facing the mistery already know the answer and everything you know. Have a "theory of mind" of both the reader and your characters when writing.

- Human Stakes: When framing some technical question as a mistery,  tell us a bit about why the hero/ human in the story cared about it. Give it some human stakes on the consequences; on why this matters. Example of good Human stales: Was that technology used in WW2 weaponry? Great, we frame it that this helped win WW2. We might exaggerate and simplify a little bit. We are learning math here, not testifying in court.

- **Show Connections and interlinks to other topics and material on the website. We don't want to just see HOW the methods / algorithms work; we want to see a big picture behind the mathematical and programming ideas that power it; we want to get new perspectives into the long intellectual history of mathematical ideas, we want to see connections to the other math topics. Consider if it would make sense for your specific article to add a dedicated "Connections" appendix where we have a bullet list of the most interesting and surprising connections of the article's topic to the topics of other articles on this website (and obvs we want hyperrefs).

- This is all with the ultimate goal of giving the reader  deeper understanding of the topic: to show not just WHAT are the mathematical ideas, tricks and concepts and programming ideas and concepts that power some method or some approach at hand, but show WHY these specific used mathematical ideas can be better choices over may be other  naive approaches and other mathematical ideas that some other people tried; or in the case where different methods would have different trade-offs give perspective and understanding into what the trade-offs are; and how the problem we are solving is approached here.
- The point is to give the perspective behind the concepts and ideas that are underlying the methods and algorithms where we show their history; and surprising connections to what we already know. We want to take a step back and look at the big picture and  In the style of James Burke BBC Connections

* Explaining and giving deep into how the math of this works;
* Taking the time to not just go through the math, but also take a look at the intuitive understanding of that math; dig into the essence of the math walkthrough; reflect on what is fundamentally going on there.   

in terms of the audience and what you can assume about what they know, think basically something that won't be out of place in the TowardsDataScience blog: you can assume familiarity with "intro to statistics" course, familiarity with writing basic numpy, pytorch code.



# Format
Heavily use formatting and add html/ js elements as tools to help make the narrative better explained and intuitive: 
* plot things 
* have interactive html visualizations; where you can browse complex-structured data by selecting things; comparing options side by side;
* mix and march different formatting and design elements to make it vary; surprise; keep it fresh.
* Bullet lists, nested bullet lists.
* Make a technical argument with a table of numbers? How about reworking it into an interactive HTML element where the user can play with some toggles, select things in choosebox and so on, and can see and compare the data points from the table in a way that would illustrate and convey the technical point /argument this data is saying in much more fun way
