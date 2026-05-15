Let's start planning a new large comic issue: research and outline the concept dependency graph we will need (the tech tree), and an outline for a multi-article issue mathematical exploration tutorial that gives a primer into the background needed to understand the kvzap paper (see attached):

- the state of the field of KV cache compression and KV cache pruning methods in which the kv zap paper was published;

- primers on the intellectual history of the ideas, tricks, insights, mathematical and ML concepts used in the  KV zap method introduced in this paper;

- Background on the heuristics and insights in the KV cache values and how the infromation is stored in them etc as pertinent to kvzap;

- Measuring and Evaluating long context LLM performance (we can link and leverage the background outlined in the comic book issue 04 "Heatmap that lied").

- Background on how GPU hardware relevant to kvzap (if any);

# tech tree of concepts
Come up with the article breakdown and concept dependency graph that we will once again call "tech tree" (like in video games):
- Look at the material (e.g. methods/ algorithms you want to explain) and break them into smaller blocks, and for each block come up with a dependency tree of mathematical ideas, mathematical insights and mathematical tricks and concepts and programming ideas that are used and needed to really undestand this block (e.g. this method that is part of the material you explain). In turn, what are mathematical ideas, tricks and concepts and programming ideas that power them? As a result, we will have a video-game style "tech tree" of these mathematical ideas. 

- For each of these concepts / "techs" in the tech tree, we want to write its own chapter, where we want to give a tutorial and an introduction into this concepts on its own; separate from the context of how it is connected to our material; we want to give an intro into the concept where we give the big picture perspective into the history of how this concept grew and was formed; how it was influenced and shaped by history, people and applications ; in the style of James Burke's BBC Connections, we would want to show how this concept would have connections to  other concepts, applications and math ideas; we want to show the connections of this concept to things and references to what the reader already knows: maybe applications, connections to nature / everyday life / history / mass culture or whatever else is approprate and we want to have some toy examples / python interactive demos and plots that show the mathematical essense and mathematical load-bearing core behind this idea.  Here we also heavily use internal hyperlinks to show the connections.

- In terms of the audience level, the audience has the mathematical  background in foundations of linear albebra, calculus, statistics, theory of probability, discrete math and so on; the audience also has at least mechanical understanding of basics of ML and architecture of LLMs. The audience is familiar with python and programming concepts. We don't have to overexplain the basics too much; and we can use appropriate terminology and leverage connections to what the reader /audience would already know. 

# The style and narrative focus

We want the exciting and engaging Radiolab / PLanet Money style of giving intro and explaining complex topics: 

- Narrate through a Mystery: Think, detective stories rather than lectures: at the beginning we are given a mistery. Throughout narration, we get more and more understanding to how the mistery can be resolved.

- Hero's Journey: Human stakes as the explanation are often framed through (simplified to the essence) struggles and breakthroughs of some scientist / inventor person.
- "Aha!" Moments: getting the reader to the moment of  epiphany where the reader gets e.g. gets to see some basic math argument in a new light or from a new perspective, or sees some non-obvious connection perhaps alongside the person from the human story.
- Anchored To The Ground-Level Entry Point. Abstract concepts are anchored to single, highly specific stories. It's better to take the time into explaining  context, stakes and background of one representative example well, than have a never ending "Also see" list of all possible ways a concept is used or manifested.
- Order of magnitude / Fermi estimates and napkin math for the key orders of magnutude;
- Concrete Metaphors and Analogies. Abstract concepts and arguments are first introduced through  examples or mapped to physical, everyday objects,  quirky situations.
- Human Emotion and Human Stakes: How did the narrative  makes the human in the story feel?
- We don't want to just see HOW the methods / algorithms work; we want to see a big picture behind the mathematical and programming ideas that power it; we want to get new perspectives into the long intellectual history of mathematical ideas.
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
