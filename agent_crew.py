from crewai import Agent, Task, Crew, Process, LLM
from config import Config

# Initialize LLM using native CrewAI class (Faster & No Warnings)
# This connects directly to your local Ollama
my_llm = LLM(
    model=f"ollama/{Config.OLLAMA_MODEL}",
    base_url=Config.OLLAMA_BASE_URL
)

def run_study_crew(query, context):
    # We combine everything into one Master Agent for maximum speed
    master_tutor = Agent(
        role='Expert Study Assistant',
        goal='Provide a clear, accurate answer based ONLY on the provided study material.',
        backstory='You are a world-class tutor. You analyze context and give high-quality answers.',
        llm=my_llm,
        allow_delegation=False,
        verbose=False
    )

    study_task = Task(
        description=f"""
        Answer the user's question using ONLY the provided context. 
        
        USER QUESTION: {query}
        
        PROVIDED CONTEXT FROM DOCUMENTS:
        {context}
        
        INSTRUCTIONS:
        1. Use the context above to answer.
        2. If the answer is not in the context, say: "I'm sorry, I couldn't find information about that in the uploaded documents."
        3. Keep the answer structured and easy to read.
        """,
        expected_output="A helpful, accurate answer based on the study material.",
        agent=master_tutor
    )

    crew = Crew(
        agents=[master_tutor],
        tasks=[study_task],
        process=Process.sequential
    )

    return crew.kickoff()