**\[TITLE\]**

**Tersoo R. Upaa Jr**					

With support from my advisor, Professor Chi Jin, and mentor Seth Karten

Monday 13, April 2026

Submitted in partial fulfillment  
of the requirements for the degree of  
Bachelor of Science in Engineering  
Department of Electrical and Computer Engineering  
Princeton University

I hereby declare that this Independent Work report represents my own work in accordance with University regulations.

I hereby declare that this Independent Work report  
does not include regulated human subjects research.

I hereby declare that this Independent Work report does not include regulated animal subjects research.

![][image1]

Tersoo R. Upaa Jr

**\[Title\]**

**Tersoo Upaa Jr					[TU8435@PRINCETON.EDU](mailto:TU8435@PRINCETON.EDU)**   
*Princeton University, Electrical & Computer Engineering*

**Abstract—**...

**Acknowledgements**  
\[Flesh out thanks section with mention of family, friends, mentors, and relationship\]  
I’d like to give a special thanks to Seth Karten for his extensive contributions towards this project and for his mentorship throughout my senior year. Though no small feat, this project has greatly benefited from his support and attention. 

I’d also like to thank SEAS and the ECE Department for their financial support.

**Outline**

Chapter 1\. Abstract

Prelude

- My history with Pokémon… what this project means to me personally… what I hope this project can mean for more people in the future.

\---------------------------------

Chapter 2\. Background **\[LATEX DRAFT DONE AND DIVERGES SOMEWHAT FROM THIS PLAN\]**

1. Introduction  
   1. …  
   2. Outline the structure of this paper → (Chapter 2, Chapter 3 \+ Chapter 4, Chapter 5 as final research component)  
      1. Be explicit in prempting that the main research components (AutoEvolve algorithm) are discussed extensively in Chapter 5  
2. Motivation & Context  
   1. The long context setting (need for additional benchmarks that test orthogonal)  
      1. Benchmarks historically served to advance the capabilities of AI  
      2. Frontier labs performing long-horizon experiments (Cursor, Anthropic) → Raises the importance of having benchmarks that natively test in this setting  
   2. Learnings from existing Pokémon playing projects/LLM Agents in Pokemon Games (Midterm)  
      1. A bit of history on the main projects and how they have captured the community  
      2. Motivating why Pokémon RPGs specifically are important (use Benchmark Paper \+ Midterm Report)  
      3. Highlight the need for standardization  
   3. Game AI Benchmarks (Benchmark)  
      1. Transition from (existing Pokemon playing projects) by highlighting the need for standardization.  
      2. Raise the importance of metrics that our benchmark tests and how orthogonal it is to other benchmarks.  
3. Related Work (in an order that properly flows)  
   1. Embodied Agents (Voyager, other examples etc)  
   2. Experiential Learning ([https://arxiv.org/abs/2403.02502](https://arxiv.org/abs/2403.02502) and other related papers)  
   3. Recursive Language Models {dealing with long context}  
   4. Self-adaptation  
      1. GEPA, SEAL (Midterm)  
      2. Skill Learning Approaches  
      3. Reset Free Optimization (Midterm)  
      4. LLM-as-a-judge (Midterm)  
   5. Meta Harnesses (MetaHarness)  
      1. [https://arxiv.org/pdf/2505.22954](https://arxiv.org/pdf/2505.22954)   
      2. [https://x.com/yoonholeee/status/2038640635482456118](https://x.com/yoonholeee/status/2038640635482456118)   
      3. [https://arxiv.org/pdf/2603.03329](https://arxiv.org/pdf/2603.03329)   
      4. Our AutoEvolve algorithm implementation is different because it (is fully online, tailored for the embodied setting)  
4. Formalization \-- for Chapter 3+ Discussion  
   1. Embodied Agent Environment (MetaHarness)  
   2. Agentic Harnesses (MetaHarness \+ Benchmark Appendix C.)  
      1. What does the agent get for free? What is it tasked with doing on its own?  
   3. … more as needed

Chapter 3\. Speedrunning Benchmark (Benchmark)

1. NeurIPS Competition Background  
   1. Benchmark is a track of the PokeAgent challenge, which was first presented as a NeurIps 2025 competition.  
   2. Evolved to a living benchmark and has further been extended to three gyms…  
2. Environment Design  
   1. The world from the perspective of the agent  
      1. Important preliminaries to help demonstrate to the reader what the agent sees which is important because it serves as ground truth for the agents access to state… including a figure here (you will have to do the mockup/draft figure for me to complete)  
   2. The world from our perspective  
      1. This is more self-evident because there is currently a figure in the appendix of the benchmark with a direct screenshot of the UI so the user can directly see the state of the game (takes inspiration from the speedrunning community) and is important for transparency.  
   3. Tutorial (tests the essential components… up to defeating roxanne) (Benchmark)  
      1. Figure 4 (Benchmark)  
      2. Text on how the best humans/Models Perform generally perform (sub 3 hours)  
      3. Simplest setting that must pass reliably before long-context ambition.  
      4. Distillation of important elements:   
         1. Simplest setting that must pass reliably before long-context ambition because it tests the basic competencies that the agent must perform consistently for the entirety of the game  
         2. Use [https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon\_Emerald](https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Emerald) as a source and highlight, in the appendix a detailed description of the full progress to the first gym.  
   4. Long Horizon (through to the third gym)... be more extensive here  
      1. New Figure, though based on a benchmark paper figure (generate good reference using nano banana pro that I will build off of)  
      2. Experiments generally complete after 12+ hours.  
      3. Distillation of important elements:   
         1. Go over how this is ample test-bed for (long-context coherence… i.e reasoning about ones own progress, decision making against adversarial opponents (battles), non-linear progression/navigation, problem solving)...   
         2. Use [https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon\_Emerald](https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Emerald) as a source and highlight, in the appendix a detailed description of the full progress through. speak to specific elements of the storyline like and how from this point there are interconnected storylines (mainling gym quest; other interconnected storylines) where the later begins to block the former. And the former can generally be completed in semi-abritrary order, making comparisons a bit trickier and vastly increasing the state of valid orderings of game progress (ordering now becomes topographical) and little decisions about the order of completion have downstream consequences on speed due to navigation time, routing constraints, resource management (pokemon number, health, levels)  
3. Speedrunning Evaluation Criteria: (use figure 24 from benchmark paper here since it distills the interconnections between the major aspects of emerald rpg through the lengs of task diversity)... The appendix can point to infrastructure (cumulative metrics) as the global metric log, including agent reasoning and tool calls.  
   1. Quantitative (Benchmark): this section will formally lay down the quantitative metrics that we think are important to capture holistically to ground our analysis of how these models perform from first principles.  
      1. Completion Percentage (captured implicitly as a dependent variable always captures milestones)  
      2. Wall-clock Time (independent variable which is a proxy for harness efficiency… recognize here that the harness construction/hardware an experiment runs on confounds this measurement, but generally, more efficient pipelines involving models that better lower latency with reasoning acumen will be more efficient along this axis.  
      3. Cumulative actions: it is important to not that at our level of granularity, we refer to actions as completed request-response pairs from the model. They can involve multiple tool calls at a given step. Models that strike a balance between chaining together multiple tool calls while respecting real-time environment dynamics (i.e the frames/time it takes to interact with items/npcs, transition between locations, etc) will be better on this metric.  
      4. Cumulative Tokens/Cost: independent variable that tracks a global view of token consumption/usage during reasoning and tool calls, as well as how this translates to costs. Harnesses with better use of caching can use more tokens at a lower cost when accounting for the choice of model… better reasoning models tend to be more expensive as well.  
   2. Qualitative (nuanced): Less structured, but gives richer analysis into the nuances of agent behaviour. And recognize just how interconnected each aspect of agent behaviour are together in affecting performance. A weaker model can, for example, be faster on a latency basis, but it it lacks proper tool use to effectively store memories or proper visual grounding, its downstream ability to develop plans may suffer and it may enter loops wondering why it cannot progress through a certain section of the game. A more powerful model can be slower on a per-turn basis, but if it effectively uses the harness, and has strong visual grounding then its plans/actions tend to be less confused, leading to more steady progress and decreasing incidences of catastrophic failure cases which can completely derail a run Here we focus on qualitative descriptions organized along the following three axes. These are best inferred by live analysis of agent thinking along side viewing the live emulated game instance, but can be done at a larger, more lossy scale via the use of llms to process the various agent logs and metrics that our benchmark tracks (run data)... Make sure that the nuances of infrastructure are discussed in the appendix, not at the level of this chapter.  
      1. Success Modes:  
         1. Descriptions of what the model is able to do, specifically, is it making effective use of the harness to properly plan for both the near term and the future, reason to effectively act, save memories, perform SLAM, visual perception, navigate its environment, solve puzzles? These are often accompanied with interesting anecdotes of agent behaviour from their reasoning traces  
      2. Failure Models:  
         1. Description of where the model is failing/struggling along the same lines as the success modes. These failure modes are interesting because in the best case, they just hurt performance, but in the worst case, they completely destroy a run. Not sure if i should mention at this level (or if this is more of a chapter ⅘ discussion by nature) but a common failure mode is lackluster visual perception… hurts localizatio and mapping to inform action, poor text processing from visual frame will lead the agent to miss out on clues that may be important narrative-wise.   
4. Speedrunning Baselines (Benchmark…): i doubt this subsection fits within the broader chapter, we can maybe include it in the appendix for now or include it in the following section since this section is about the benchmark itself and im not discussing results.  
   1. Human baselines  
   2. Harness vs. Model Capability 

\-------------------------------------------------------  
\-------------------------------------------------------  
\-------------------------------------------------------  
\-------------------------------------------------------  
Chapter 4\. PokeAgent Baseline Harness

- Here we discuss the development of the expert harness and many learnings along the way that have bearing both on the domain of pokemon rpgs, but also beyond to long-horizon agent behavior in embodiedsettings.  
1. Early Development  
   1. Competition Era (v3.0.0-local branch of github repo)  
      1. Early insights  
      2. Description: 4 modules, only short term memory of previous actions up to 20 steps. no planning/advanced mapping/pathfinding tools, simple Analyze, Plan, reasoning, Action pipeline grounded in the principles of agentic prompt engineering source/papwer \+ Draft Figure of this simple starting point  
         1. The state of the repo for the pokeagent competition era agent is in the branch v3.0.0-local of the github repo. This is the simplest 4-module architecture for which the PokeAgent Challenge Retrospective refers to as the PokeAgent Simple Baseline for the competition in late 2025\.   
         2. Use nano banana pro to generate the draft figure for this simple 4 module agent.  
   2. Towards Subagents (this should be after Early Performance section b\! Since the learnings from the bench mark papers agents results spurred the completion of the expert harness.)  
      1. Refer back to Motivation & Context, →   
         1. Model capability has increased substantially through the duration of this project (Gemini 2.5 vs Gemini 3 multi-modal reasoning alone showed improvement)  
         2. CLI Harnesses have been rapidly evolving, notably by their abstractions for planning, tool use, and EVENTUALLY… subagents\!  
      2. Description \+ Figure of the intermediate baseline (Midterm Paper)   
2. Early Performance  
   1. Competition Era (PokeAgent Simple) Baseline \+ Results from Competitors (Benchmark)  
      1. Results Figure 7 from benchmark paper: Competitors vs. PokeAgent (Benchmark)  
      2. Discussion → weakest pokeagent simple baseline w/ little abstractions vs. Evelord (intermediate form of Pokeagent baseline with a non-categorized objective system \+ porymap decomp data) vs. hyper-specific well-performing methods with little generalization.   
         1. Notes on the simplest baseline that spurred development:  
            1. The release of Gemini 3 models led to a large increase in performance, specifically around visual processing, as actions seemed to be more visually grounded, rather than overreliance on textual state (NOTE: do some research here to cite the Gemini 3 paper from DeepMind to better position this discussion in the technical literature). This helped with objectives requiring fine movement, like the setting of the clock in the player's bedroom, which was previously very hard for Gemini 2.5 models and increased consistency. This not only underscored the importance of using a performant base model, but also stressed how effective multi-modal reasoning was. The earliest versions of the   
            2. 4 module architecture was poor on both a step/time level… high inference cost for each action.  
            3. Lacked a lot of coherence and would frequently get confused about game progress. Because the tutorial setting was fairly linear, this was an effective, but still slow strategy. This  version of the agent would often infer its next action based on the game screen/its location and its world knowledge to make slow progress.   
         2. Notes from evelord that spurred development for version of PokeAgent testing in retrospective paper:  
            1. Objective system was a big step in developing consistency in execution. By isolating planning and action phases, the agent could develop plans both dynamically or we could prompt it to generate a full sequence of the at once using the same walkthrough source.  
            2. Added a knowledge base as well as summary tools (precusor to later Memory.json abstraction) where the agent could store discoveries in a list to help it better localize its performance when generating plans. The objective system and knowledge base greatly helped in consistent performance.  
            3. Here are some notes about persistent failure cases that I noticed at this stage:  
               1. navigating route 104  
               2. \- Agent lacks the ability to critique/improve objectives when stuck (i have to manually tell it to navigate to a different warp, and even then it takes a while)  
               3. \- agent is very sensitive to slight nuances in the naming of objectives (if i have route\_104\_south as an objective, It tries to navigate to the southern part of route 104   
               4. \- Need to improve pathfinding to respect tiles only walkable in a certain direction  
               5. \- HARD: Need to prompt agent to — in the case it gets sent back to a pokemon center — to review the objectives it has completed if it needs help navigating back to its original location  
               6. \- Agent struggles to navigate on very large maps with many warps  
               7. \- NOTE: Water is currently unwalkable, will need to fix once agent can surf  
               8. \- Agent needs more prompting to distinguish upper and lower warps to route 104 when in petalburg woods  
               9. Rustburo city gym (moving trainers messes up pathfinding without npc detection and the agent overrelies on tools that it assume can work. It didnt revert to manual navigation and focus more on the visual frame, instead i had to manually perform a stochastic fix.)  
               10. \- fixing via increased variance in paths  
         3. This can pull from the summaries each winning battling track team submitted. You can discuss these disparate methods at a high level here (as well as their learnings) and point to a new section in the appendix that takes from the appendix E.2 of the PokeAgent Challenge retrospective verbatim (including figures since these were direct submissions from competitors). Make sure to cite the retrospective paper here when doing this since i didnt personally work on collecting and organizing group submissions.  
   2. Results from the Benchmark Paper (Benchmark)  
      1. Results Figure from benchmark: Speedrunning Harness vs. Model (Use figure 5 from PokeAgent Challenge Retorpsective/Benchmark paper as a draft for now that I will edit)  
      2. Discussion  
         1. Comparing SOTA models… where Gemini outperforms because it shines is in visual reasoning (even though Claude/GPT are generally considered better models \[cite\])... underscores the importance of multi-modal reasoning and perception. Gemini also seemed to have better world knowledge of Pokemon, which helped it. For example, gemini models would almost always pick mudkip and its reasoning traces would explicitly reveal that this was because it knew that mudkip was a more effective starter. GPT/Claude models were less consistent in this regard, affecting downstream knowledge. While not directly relevant to the scope of this paper, which doesnt do training to update model weights, it nonetheless suggests that explicit finetuning/training on a particular domain is important.  
         2. Highlight the promise of CLI agents meant to solve coding tasks… they share a lot of the same architectural primitives that generalize for other domains.  
            1. Qualitative observations of how they use native planning systems, though often at an unsuitable level of granularity (not specific enough to provide meaningful steering, which is important).  
            2. Caching (on the order of \~200k tokens before compaction) \+ strong Pokémon world knowledge enables effective, in-context planning without persistent use of planning abstractions.  
            3. Proprietary problem-solving scaffolding allows them to explore hypotheses directly in an embodied setting, and use research tools online to help redi rect when they get stuck… AT THE COST OF SIGNIFICANTLY HIGHER TOKEN BURN

         3. Our strongest baseline H\_expert (developed for AutoEvolve comparisons in chapter 5)... (modern tersoo-dev-2-research branch)  
            1. Fully fleshed out system design figure ( use the Benchmark paper figure 14: pokeagent multi-agent architecture as a draft that i will refine later)  
               1. Orchestrator \+ subagent paradigm (context management problem) \+ Useful abstractions to process session history/long-term memory (coherence and localization problem)  
               2. This new system is engineered to better suit the long-term nature of the PokeAgent challenge (*Long Horizon* in environment design). As we will see, it takes a hit in the absolute step-wise efficiency metric because of all of the context switching. Still, it better prevents catastrophic forgetting/erroneous reasoning, which can halt progress entirely in the long run. In short, it provides a more powerful reasoning harness.  
            2. Discuss choice of Clear abstractions for  
               1. Trajectory Window  
                  1. RLM inspired… akin to a session log. Our custom agents manually process these large streams of context and pass their analysis to the orchestrator.  
                  2. Context Management \+ maintaining coherence.  
                  3. Importantly, we provide a primitive for processing the trajectory window that will become important when discussing H\_min/H\_auto… The subagents that operate on the trajectory window are the 1-step subagents.  
               2. Memory:   
                  1. Short-term (tool call results in 20-step window \+ memory/skill/subagent overview) vs long-term (accessible via CRUD) @ higher latency which corresponds to mental effort  
               3. Skills:   
                  1. Discuss the particulars (modfiable via basic crud operations. Overview is given as short term memory and specific skills can be read via cru  
               4. Subagents   
                  1. We provide powerful custom agents (to help decompose certain aspects of the game and also to help in context management to the main orchestrator).   
                  2. Importantly, we provide infrastructure for a general subagent primitive exposed as a tool call (will become important when discussing H\_min/H\_auto in chapter 5\)  
               5. Sandbox/Code  
                  1. Discuss the particulars

Chapter 5\. AutoEvolve

1. Beyond Pokemon  
   1. Transition from Chapter 4 → Chapter 5 by position auto-evovle as a domain-agnostic evolutionary loop, which needs certain primitives (modular agent framework, sandboxing, and a global session log, general subagents) and that leverages llm-as-a-judge to in a reset-free manner.  
   2. Describe the AutoEvolve algorithm here and add a draft figure that I will flesh out which shows the primary agent loop and auto-evolve as a complementary evolutionary loop.  
   3. Auto-evolve is inherently domain agnostic  
2. Methodology  
   1. Algorithm (write the algorithm as a figure here with function calls as necessary)...  
   2. Experimental Design  
      1. 3 settings: Baseline \+ Baseline w/ AutoEvolve \+ Fully Custom PokeAgent (3 trials each)  
         1. Test gemini 3.1-pro-preview  
         2. Test gemini 3-flash-preview  
      2. Bootstrapped Transfer Learning (3 trials each)  
         1. Equip a weaker orchestrator with the Auto-evolved harness of a more powerful orchestrator \+ supervisor combination  
   3. Metrics  
      1. Refer back to standard evaluation criteria… Each experiment setting will involve a general discussion of qualitative observations  
3. Results (For qualitative metrics, we touch on the highlights of Appendix 2.d)  
   1. 3 Settings results  
      1. Speedrunning Evlautaiton Metris  
      2. Qualitative Results  
   2. Bootstrapped Transfer Learning Results  
      1. Speedrunning Evlautaiton Metris  
      2. Qualitative Results  
4. Discussion  
   1. How did auto-evolve in comparison to simplest baseline and fully custom PokeAgent.  
      1. What were the main failure/success cases… why did they perform that way and how does the literature support our argument?  
   2. What are the implications of the results (can we extend this method to other domains)  
      1.   
   3. … more as needed

Chapter 6\. Outlook

1. Conclusion  
   1. What Have I Developed  
      1. Benchmark  
      2. Custom VLM-agent Architecture w/ powerful abstractions for embodied agent settings  
      3. AutoEvolve algorithm for self-adaptation of model harnesses via experiential learning  
   2. What Have I Shown and Why is it Promising  
2. Future Work  
   1. Model Distillation to train opensource models on a harness learned by a more powerful model  
      1. Interesting implication sin the industry… Teacher model via auto-evolve   
   2. Full Game evaluations  
   3. Extensions to other domains (coding, robotics, etc)

Chapter 7\. Ancillary Sections

1. Bibliography  
2. Appendix  
   1. Nuanced discussion of Environment and System Details beyond the purview of the main paper itself. (Benchmark Appendix, organized as necessary)  
      1. Also include discussion of porymap decompilation project data being used to supplement the state information with npc/object/warp locations, which is an important state-level consideration that has direct implications on agent performance.  
   2. **Relevant appendix section(s) for chapter 4 content.**  
   3. Example Prompts used for ablations in chapter 5 Studies  
   4. Gameplay demonstration (probably to showcase chapter 5 results… these will link to youtube videos that I create)  
   5. Full Memory/Skill/Subagent JSON files Comparisons across AutoEvolve Experiments to showcase relevant results

[image1]: <data:image/png;base64,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>