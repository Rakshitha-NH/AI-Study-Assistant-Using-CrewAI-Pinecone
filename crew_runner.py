from crewai import Crew, Process
from tasks import create_tasks
from agents import research_agent, analysis_agent, review_agent

def ask_study_assistant(question: str):
    tasks = create_tasks(question)
    
    crew = Crew(
        agents=[research_agent, analysis_agent, review_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    # Execute the workflow defined in the diagram
    result = crew.kickoff()
    return result