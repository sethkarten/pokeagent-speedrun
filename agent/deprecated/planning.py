import logging
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary
from agent.system_prompt import system_prompt

# Set up module logging
logger = logging.getLogger(__name__)

def planning_step(memory_context, current_plan, slow_thinking_needed, state_data, vlm):
    """
    Decide and update your high-level plan based on memory context, current state, and the need for slow thinking.
    Returns updated plan.
    """
    # Get formatted state context
    state_context = format_state_for_llm(state_data)
    state_summary = format_state_summary(state_data)
    
    logger.info("[PLANNING] Starting planning step")
    logger.info(f"[PLANNING] State: {state_summary}")
    logger.info(f"[PLANNING] Slow thinking needed: {slow_thinking_needed}")
    
    # Check if current plan is accomplished
    if current_plan:
        plan_check_prompt = f"""
        ★★★ COMPREHENSIVE GAME STATE DATA ★★★
        
        {state_context}
        
        ★★★ PLAN ASSESSMENT TASK ★★★
        
        You are the agent playing Pokemon Emerald. Assess your current situation and plan progress.
        
        Current Plan: {current_plan}
        Memory Context: {memory_context}
        
        Considering your current location, pokemon party, money, traversability, and recent actions:
        Have you accomplished your current plan? Answer YES or NO, and explain briefly.
        
        Consider these factors:
        - Did you reach your target location?
        - Did you complete the intended battle/gym challenge?
        - Did you acquire the needed pokemon/items?
        - Are you stuck due to terrain or party status?
        - Do you need to adapt due to wild encounters or water obstacles?
        """
        plan_status = vlm.get_text_query(system_prompt + plan_check_prompt, "PLANNING-ASSESSMENT")
        if "yes" in plan_status.lower():
            current_plan = None
            logger.info("[PLANNING] Current plan marked as completed")
    
    # Generate new plan if needed
    if current_plan is None or slow_thinking_needed:
        planning_prompt = f"""
        ★★★ COMPREHENSIVE GAME STATE DATA ★★★
        
        {state_context}
        
        ★★★ STRATEGIC PLANNING TASK ★★★
        
        You are the agent playing Pokemon Emerald with a speedrunning mindset. Create an efficient strategic plan.
        
        Memory Context: {memory_context}
        
        Analyze your situation and create a strategic plan:
        
        1. IMMEDIATE GOAL: What should you focus on right now? Consider:
           - If in battle: What's your battle strategy based on pokemon HP/levels?
           - If on map: Navigate efficiently using traversability data
           - If in menu/dialogue: How to progress efficiently?
           - Do you need to heal pokemon at Pokemon Center?
           - Are there terrain obstacles (water, blocked paths) to navigate?
        
        2. SHORT-TERM OBJECTIVES (next few actions):
           - Specific steps to achieve your immediate goal
           - Account for your current pokemon party health and levels
           - Consider terrain: avoid/seek tall grass, navigate around obstacles
           - Money management for items/healing
        
        3. LONG-TERM STRATEGY:
           - How does this fit into beating the game quickly?
           - What gym leader or major milestone to target next?
           - Pokemon catching/training priorities based on current party
           - Route optimization considering terrain types
        
        4. EFFICIENCY NOTES:
           - How to minimize backtracking using map layout
           - Shortcuts or sequence breaks considering terrain
           - Wild encounter management (avoid/seek based on needs)
        
        Format as a clear, actionable plan focusing on speed and efficiency.
        """
        current_plan = vlm.get_text_query(system_prompt + planning_prompt, "PLANNING-CREATION")
        logger.info("[PLANNING] New plan created")
    
    logger.info(f"[PLANNING] Final plan: {current_plan[:300]}..." if len(current_plan) > 300 else f"[PLANNING] Final plan: {current_plan}")
    return current_plan 