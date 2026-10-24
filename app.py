#---CrewAI projects served by FastAPI: ten routes, /agent1 to /agent10.
#---Open "/" for the website, "/docs" for the API. Every route takes a valid input and returns a valid output.
#---Each project is written in the order of the seven steps in the CrewAI notes:
#---Step 1 Setup, 2 LLM, 3 Agent (State in a Flow), 4 Task (Start), 5 Crew (Steps), 6 Run, 7 Result.
import os
import re
import csv
import io
import json
import time
import tempfile
import threading
import functools
import html
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, HttpUrl
import pandas as pd
from ddgs import DDGS

#---Step 1: Setup(import crewai and set the API key)
from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import Flow, listen, or_, and_, router, start
from crewai.flow.human_feedback import human_feedback
from crewai.flow.async_feedback import HumanFeedbackProvider, HumanFeedbackPending
from crewai.tools import tool
from crewai_tools import ScrapeWebsiteTool, FileReadTool
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY","true")
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY","")      # set on Render under Environment (userdata.get in Colab)

#---Step 2: LLM(choose the model and its settings)
GEMINI_MODEL=os.environ.get("GEMINI_MODEL","gemini/gemini-3.1-flash-lite-preview")
llm=LLM(model=GEMINI_MODEL,api_key=GEMINI_API_KEY,temperature=0.4) if GEMINI_API_KEY else None
#---end of steps 1 and 2

app=FastAPI(title="CrewAI projects",description="Ten CrewAI Crews and Flows. One route each: /agent1 to /agent10. Open / for the website.",version="2.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
AGENT_SETTINGS=dict(verbose=False,allow_delegation=False,max_iter=6,memory=False)    # the Agent settings from the notes


#---Helpers used by every route
def busy(error):
  text=str(error).lower()
  return any(word in text for word in ("503","unavailable","high demand","overloaded","429","resource_exhausted"))

def ask(prompt,Model=None):
  #---one direct model call, like llm.call(prompt).strip() in the notebook
  #---with a Pydantic Model the answer is validated against it
  for attempt in range(3):
    try:
      if Model is None:
        return str(llm.call(prompt)).strip()
      raw=llm.call(messages=[{"role":"user","content":prompt}],response_model=Model)
      if isinstance(raw,Model):
        return raw
      text=raw if isinstance(raw,str) else json.dumps(raw)
      first,last=text.find("{"),text.rfind("}")
      return Model.model_validate_json(text[first:last+1] if first>=0 else text)
    except Exception as error:
      if busy(error) and attempt<2:
        time.sleep(15)
        continue
      raise

slots=threading.BoundedSemaphore(2)       # two runs at a time, so the free request limit is respected

def guarded(function):
  #---waits for a free slot, retries when Gemini is busy, and turns any failure into a clear JSON error
  @functools.wraps(function)
  def wrapper(*args,**kwargs):
    if llm is None:
      raise HTTPException(status_code=503,detail="GEMINI_API_KEY is not set on the server.")
    if not slots.acquire(timeout=300):
      raise HTTPException(status_code=429,detail="The server is busy with other runs. Try again in a minute.")
    try:
      for attempt in range(3):
        try:
          return function(*args,**kwargs)
        except HTTPException:
          raise
        except Exception as error:
          if busy(error) and attempt<2:
            time.sleep(20*(attempt+1))
            continue
          raise
    except HTTPException:
      raise
    except Exception as error:
      text=str(error).lower()
      if "api key" in text or "permission_denied" in text or "401" in text or "403" in text:
        raise HTTPException(status_code=502,detail="The Gemini API key was rejected. Check GEMINI_API_KEY.")
      if busy(error):
        raise HTTPException(status_code=503,detail="The Gemini model is busy or its request limit is reached. Try again shortly.")
      raise HTTPException(status_code=500,detail="The run failed: "+str(error)[:300])
    finally:
      slots.release()
  return wrapper

def crew_trail(crew):
  #---which agent did each task, in order
  return [t.output.agent for t in crew.tasks if getattr(t,"output",None) is not None and getattr(t.output,"agent",None)]

def flow_trail(flow):
  #---which Flow methods ran, in order
  return [m["method"] for m in getattr(flow,"_method_outputs",[])]

@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
  """Search the web using DuckDuckGo and return titles, links and summaries."""
  try:
    results=DDGS().text(query,max_results=5)
    lines=[f"Title: {r.get('title','')}\nLink: {r.get('href','')}\nSummary: {r.get('body','')}" for r in results]
    return "\n\n".join(lines) if lines else "No results found."
  except Exception as error:
    return f"Search failed: {error}"

def make_writer(folder):
  #---a file writer that can only write inside this request's own folder
  @tool("File Writer")
  def file_writer(filename: str, content: str) -> str:
    """Save text to a file. Arguments: filename (for example report.md) and content."""
    name=os.path.basename(filename) or "output.md"
    with open(os.path.join(folder,name),"w",encoding="utf-8") as f:
      f.write(content)
    return f"Saved {name}."
  return file_writer

def write_file(folder,filename,text):
  #---FileWriterTool in plain Python: save text inside this request's own folder
  with open(os.path.join(folder,os.path.basename(filename)),"w",encoding="utf-8") as f:
    f.write(text)

def append_row(folder,filename,header,row):
  #---add one row to a CSV file, and write the header first when the file is new
  path=os.path.join(folder,filename)
  new=not os.path.exists(path)
  with open(path,"a",encoding="utf-8",newline="") as f:
    writer=csv.writer(f)
    if new:
      writer.writerow(header)
    writer.writerow(row)

def saved(folder,filename,fallback):
  path=os.path.join(folder,filename)
  if os.path.exists(path) and os.path.getsize(path)>20:
    return open(path,encoding="utf-8").read()
  return fallback

#---Sample inputs: they are the defaults of every route, so an empty request is still a valid request
SAMPLES={'topic': 'AI in farming',
 'competitors': ['https://www.python.org', 'https://nodejs.org', 'https://go.dev'],
 'your_product': 'A training course on backend programming',
 'transcript': 'Weekly product meeting, Tuesday 14 October. Present: Meera (product), Arjun (engineering), Sana '
               '(design), Dev (support).\n'
               '\n'
               'Meera: Thanks all. Three items: the checkout redesign, the mobile crash reports, and the January '
               'launch date.\n'
               'Sana: The checkout redesign is ready for review. I need engineering feedback by Friday so that the '
               'handover can happen next Monday.\n'
               'Arjun: I will review it by Thursday evening. One concern: the new address form needs a postcode '
               'lookup, and we have no vendor for that yet.\n'
               'Meera: Let us decide that now. Dev, you have used two lookup vendors in support. Which would you '
               'recommend?\n'
               'Dev: The cheaper one had frequent timeouts. I recommend the second, at about forty dollars a month.\n'
               'Meera: Agreed. We will go with the second vendor. Arjun, please create the account and share the key '
               'by next Wednesday.\n'
               'Arjun: Understood.\n'
               'Dev: On crashes, there are about thirty reports a week on Android 12, all from the payment screen.\n'
               'Arjun: That matches a library upgrade two weeks ago. I will roll the library back this week and open '
               'a ticket to upgrade it properly next sprint.\n'
               'Meera: Good. Dev, please send Arjun the five most detailed crash reports today.\n'
               'Dev: Will do.\n'
               'Meera: On the launch date: marketing wants 12 January, but we do not have the postcode lookup tested '
               'until the end of December.\n'
               'Sana: I would rather move the launch to 19 January than ship the form untested.\n'
               'Meera: Decision: the launch moves to 19 January. I will tell marketing tomorrow.\n'
               'Meera: Anything else? No. Thank you all.',
 'csv': 'date,region,product,units,unit_price\n'
        '2026-09-01,North,Backpack,27,38\n'
        '2026-09-01,South,Notebook,20,4.5\n'
        '2026-09-01,East,Backpack,25,38\n'
        '2026-09-01,West,Notebook,60,4.5\n'
        '2026-09-08,North,Desk lamp,27,22\n'
        '2026-09-08,South,Backpack,31,38\n'
        '2026-09-08,East,Desk lamp,17,22\n'
        '2026-09-08,West,Backpack,36,38\n'
        '2026-09-15,North,Desk lamp,25,22\n'
        '2026-09-15,South,Desk lamp,25,22\n'
        '2026-09-15,East,Notebook,16,4.5\n'
        '2026-09-15,West,Headphones,63,64\n'
        '2026-09-22,North,Notebook,33,4.5\n'
        '2026-09-22,South,Headphones,19,64\n'
        '2026-09-22,East,Backpack,26,38\n'
        '2026-09-22,West,Desk lamp,38,22\n'
        '2026-09-29,North,Notebook,47,4.5\n'
        '2026-09-29,South,Backpack,20,38\n'
        '2026-09-29,East,Desk lamp,22,22\n'
        '2026-09-29,West,Notebook,56,4.5\n'
        '2026-10-06,North,Backpack,37,38\n'
        '2026-10-06,South,Notebook,34,4.5\n'
        '2026-10-06,East,Notebook,33,4.5\n'
        '2026-10-06,West,Backpack,36,38\n'
        '2026-10-13,North,Desk lamp,41,22\n'
        '2026-10-13,South,Headphones,28,64\n'
        '2026-10-13,East,Notebook,20,4.5\n'
        '2026-10-13,West,Backpack,31,38\n'
        '2026-10-20,North,Notebook,45,4.5\n'
        '2026-10-20,South,Desk lamp,28,22\n'
        '2026-10-20,East,Headphones,22,64\n'
        '2026-10-20,West,Backpack,46,38',
 'tickets': [{'label': 'Billing',
              'ticket_id': 'T-1001',
              'customer_name': 'Asha Rao',
              'issue_description': 'Charged twice for October. Hello, my card was charged 29.00 twice on 3 October '
                                   'for the Pro plan. Please refund one of the charges. The order number is 88213.'},
             {'label': 'Account access',
              'ticket_id': 'T-1002',
              'customer_name': 'Ravi Teja',
              'issue_description': "Cannot sign in. Since this morning I get 'invalid code' when I enter the "
                                   'two-factor code, and I no longer have my old phone. How do I get back into my '
                                   'account?'},
             {'label': 'Other',
              'ticket_id': 'T-1003',
              'customer_name': 'Meera Nair',
              'issue_description': 'Export to CSV. Could the reports page include an option to export the table as '
                                   'CSV? We copy the numbers into a spreadsheet every week.'}],
 'job': 'Junior Data Analyst. We need someone who can clean and analyse sales data with SQL and Python (pandas), '
        'build clear charts, and explain findings to non-technical colleagues. Required: SQL, Python, Excel, written '
        'English. Preferred: Tableau or Power BI, a statistics course. You will report to the head of operations.',
 'resumes': [{'name': 'Anita Rao',
              'text': 'BSc Statistics, 2025. Internship at a retail chain: wrote SQL queries to find slow-moving '
                      'stock and built a weekly pandas report that replaced a manual Excel process. Final-year '
                      'project: forecasting store sales with Python. Skills: SQL, Python (pandas, matplotlib), '
                      'Excel, Power BI. Completed a university statistics course with distinction.'},
             {'name': 'Karan Mehta',
              'text': 'BCom, 2024. Two years in retail sales. Skills: Excel (pivot tables), basic SQL from an online '
                      'course, customer service. Has not used Python. Interested in moving into analytics.'},
             {'name': 'Leila Haddad',
              'text': 'MSc Data Science, 2026. Thesis on churn prediction using Python and scikit-learn. Part-time '
                      'analyst for a charity: SQL dashboards in Tableau, monthly reports for trustees. Skills: SQL, '
                      'Python, Tableau, Excel, clear written English.'}],
 'issues': [{'label': 'Bug',
             'title': 'App crashes when saving a project with an emoji in the name',
             'body': "Steps: create a project, rename it to 'Launch 🚀', press Save. The page goes blank and the "
                     "console shows 'Invalid string length'. This happens in Chrome 126 on Windows. It worked last "
                     'week.'},
            {'label': 'Feature',
             'title': 'Add dark mode',
             'body': 'Many of us work late and the white background is tiring. Could there be a dark theme that '
                     'follows the system setting?'},
            {'label': 'Question',
             'title': 'How do I run the crew from a script?',
             'body': 'I followed the tutorial in the notebook. How do I run the same crew from a normal Python file '
                     'instead?',
             'docs_url': 'https://docs.crewai.com/en/introduction'}],
 'article': 'Why small teams should write down how they decide\n'
            '\n'
            'Most small teams make decisions in conversation, and forget them within a month. Three months later, '
            'someone asks why the team chose one supplier over another, and nobody can answer with confidence.\n'
            '\n'
            'A decision record fixes this at low cost. It is a short page with four parts: the question, the options '
            'considered, the choice made, and the reason. It takes ten minutes to write and is kept in a folder '
            'everyone can open.\n'
            '\n'
            'The benefit is not bureaucracy. When a new colleague joins, they can read the last twenty decisions in '
            'an afternoon and understand how the team thinks. When circumstances change, the team can see which '
            'reasons no longer hold, and reverse the choice deliberately instead of by accident.\n'
            '\n'
            'Teams that adopt the habit report two effects. Meetings become shorter, because the record shows what '
            'was already settled. Disagreements become more specific, because people must state a reason in '
            'writing.\n'
            '\n'
            'The practice only works if records are short. A record longer than one page is rarely read, and so '
            'rarely written. Start with the next decision that costs money or takes more than a week to reverse.',
 'leads': [{'label': 'Software company',
            'name': 'Python Software Foundation',
            'website': 'https://www.python.org',
            'offer': 'Developer training courses for engineering teams of 20 to 200 people'},
           {'label': 'Small shop',
            'name': 'Example Domain',
            'website': 'https://example.com',
            'offer': 'Developer training courses for engineering teams of 20 to 200 people'}],
 'report': {'topic': 'Remote work and team productivity', 'audience': 'Operations managers at mid-sized companies'}}


# =====================================================================================
# 1. Blog Post Crew - Crew, sequential process
# =====================================================================================
class BlogRequest(BaseModel):
  topic: str=Field(default=SAMPLES["topic"],min_length=3,max_length=120)

class BlogResponse(BaseModel):
  topic: str
  blog_post: str
  trail: List[str]

@app.post("/agent1",response_model=BlogResponse,tags=["Agents"],summary="Blog Post Crew")
@guarded
def agent1(req: Optional[BlogRequest]=None):
  """Researcher, Writer and Editor produce a 600-word blog post. Sequential process."""
  req=req or BlogRequest()
  folder=tempfile.mkdtemp()

  #---Step 3: Agent(define who does the work)
  researcher=Agent(role="Researcher",goal="Find accurate, current facts about the topic and record where each came from",
                   backstory="You research carefully and never state a fact that a source did not give you.",tools=[duckduckgo_search],llm=llm,**AGENT_SETTINGS)
  writer=Agent(role="Writer",goal="Write a clear 600-word blog post using only the research notes",
               backstory="You write plain, well-structured articles for a general reader.",llm=llm,**AGENT_SETTINGS)
  editor=Agent(role="Editor",goal="Correct the tone and remove any claim the research does not support",
               backstory="You are a strict editor. You compare each claim with the notes and delete what is not supported.",tools=[make_writer(folder)],llm=llm,**AGENT_SETTINGS)

  #---Step 4: Task(define what needs to be done)
  task1=Task(description="Search the web for: {topic}. List 6 to 8 key facts. Give the source address beside each fact. Use only what the searches return.",
             expected_output="A bullet list of facts, each with a source address.",agent=researcher)
  task2=Task(description="Write a blog post of about 600 words on {topic}. Use only facts from the research notes. Begin with a title and end with a Sources section.",
             expected_output="A blog post in Markdown with a title and a Sources section.",agent=writer,context=[task1])
  task3=Task(description="Check every claim in the draft against the research notes. Remove or correct any claim they do not support, and improve the tone. "
                         "Save the final text with the File Writer tool using the filename blog_post.md. Then return the final text.",
             expected_output="The final blog post in Markdown.",agent=editor,context=[task1,task2])

  #---Step 5: Crew(combine the agents and tasks)
  crew=Crew(agents=[researcher,writer,editor],tasks=[task1,task2,task3],process=Process.sequential,verbose=False)

  #---Step 6: Run(start the crew and let it work)
  result=crew.kickoff(inputs={"topic":req.topic})

  #---Step 7: Result(read the final output)
  return BlogResponse(topic=req.topic,blog_post=saved(folder,"blog_post.md",result.raw),trail=crew_trail(crew))


# =====================================================================================
# 2. Competitor Analysis Crew - Crew, hierarchical process (manager_llm)
# =====================================================================================
class CompetitorRequest(BaseModel):
  competitors: List[HttpUrl]=Field(default_factory=lambda:[HttpUrl(u) for u in SAMPLES["competitors"]],min_length=1,max_length=3)
  your_product: str=Field(default=SAMPLES["your_product"],min_length=3,max_length=200)

class CompetitorResponse(BaseModel):
  report: str
  trail: List[str]

@app.post("/agent2",response_model=CompetitorResponse,tags=["Agents"],summary="Competitor Analysis Crew")
@guarded
def agent2(req: Optional[CompetitorRequest]=None):
  """A manager delegates scraping, pricing and news work to specialists and returns a SWOT report. Hierarchical process."""
  req=req or CompetitorRequest()
  folder=tempfile.mkdtemp()

  #---Step 3: Agent(define who does the work)
  scraper=Agent(role="Site Scraper",goal="Read each competitor page and report what is visibly on it",
                backstory="You read pages exactly and say so when a page returns little or no text.",tools=[ScrapeWebsiteTool()],llm=llm,**AGENT_SETTINGS)
  pricing=Agent(role="Pricing Analyst",goal="Extract plan names, prices and limits from scraped text",
                backstory="You report only prices that appear in the text. You write 'not found' otherwise.",llm=llm,**AGENT_SETTINGS)
  news=Agent(role="News Analyst",goal="Find recent announcements and press about each competitor",
             backstory="You report dated items with links, and nothing you cannot link to.",tools=[duckduckgo_search],llm=llm,**AGENT_SETTINGS)
  strategist=Agent(role="Strategist",goal="Combine the findings into a SWOT table for our product",
                   backstory="You compare evidence and avoid speculation.",tools=[make_writer(folder)],llm=llm,**AGENT_SETTINGS)

  #---Step 4: Task(define what needs to be done; no agent is named, the manager decides)
  task1=Task(description="Use ScrapeWebsiteTool on each of these addresses: {competitors}. For each, state what the company offers and quote the main headings. If a page returns little text, write 'page not readable'.",
             expected_output="A short factual profile per competitor.")
  task2=Task(description="From the scraped text only, list plan names, prices and limits for each competitor. Write 'not found' where the text does not say.",
             expected_output="A pricing table per competitor.",context=[task1])
  task3=Task(description="Search for recent announcements, launches or press about these companies: {competitors}. Return dated items with links.",
             expected_output="A dated list of news items with links.")
  task4=Task(description="Our product is: {product}. Using the profiles, prices and news, build a SWOT table for our product against these competitors. "
                         "Save the report with the File Writer tool using the filename report.md, then return it.",
             expected_output="A Markdown report with a summary per competitor, a pricing comparison, recent news and a SWOT table.",context=[task1,task2,task3])

  #---Step 5: Crew(a manager model leads the agents)
  crew=Crew(agents=[scraper,pricing,news,strategist],tasks=[task1,task2,task3,task4],process=Process.hierarchical,manager_llm=llm,verbose=False)

  #---Step 6: Run(start the crew and let it work)
  result=crew.kickoff(inputs={"competitors":"\n".join(str(u) for u in req.competitors),"product":req.your_product})

  #---Step 7: Result(read the final output)
  return CompetitorResponse(report=saved(folder,"report.md",result.raw),trail=crew_trail(crew))


# =====================================================================================
# 3. Meeting Notes Crew - Crew, parallel tasks joined through context
# =====================================================================================
class MeetingRequest(BaseModel):
  transcript: str=Field(default=SAMPLES["transcript"],min_length=200,max_length=20000,description="The full meeting transcript")

class ActionItem(BaseModel):
  owner: str=Field(description="Person responsible, as named in the transcript")
  task: str=Field(description="What must be done")
  due: str=Field(description="Due date or day as stated, or 'not stated'")

class ActionList(BaseModel):
  items: List[ActionItem]

class MeetingResponse(BaseModel):
  notes: str
  actions: List[ActionItem]
  trail: List[str]

@app.post("/agent3",response_model=MeetingResponse,tags=["Agents"],summary="Meeting Notes Crew")
@guarded
def agent3(req: Optional[MeetingRequest]=None):
  """Summary, action items and decisions are found in parallel, then assembled into notes."""
  req=req or MeetingRequest()
  folder=tempfile.mkdtemp()
  path=os.path.join(folder,"transcript.txt")
  with open(path,"w",encoding="utf-8") as f:
    f.write(req.transcript)
  note="You report only what the transcript says. You never invent names, dates or decisions."

  #---Step 3: Agent(one agent cannot run two tasks at the same moment, so each parallel task has its own)
  summarizer=Agent(role="Summarizer",goal="Summarise the meeting faithfully",backstory=note,tools=[FileReadTool()],llm=llm,**AGENT_SETTINGS)
  action_finder=Agent(role="Action Finder",goal="Find every action item with its owner and due date",backstory=note,tools=[FileReadTool()],llm=llm,**AGENT_SETTINGS)
  decision_logger=Agent(role="Decision Logger",goal="Record the decisions that were actually made",backstory=note,tools=[FileReadTool()],llm=llm,**AGENT_SETTINGS)
  formatter=Agent(role="Formatter",goal="Assemble the three results into one clean notes file",
                  backstory="You combine given material without changing its meaning.",tools=[make_writer(folder)],llm=llm,**AGENT_SETTINGS)

  #---Step 4: Task(three tasks run side by side with async_execution=True)
  where=f"Read the file at {path} with the file reading tool. "
  task1=Task(description=where+"Write a summary of the meeting in exactly five lines.",expected_output="Five lines.",agent=summarizer,async_execution=True)
  task2=Task(description=where+"List every action item with its owner, the task and the due date. If no date was stated, write 'not stated'.",
             expected_output="A validated list of action items.",agent=action_finder,async_execution=True,output_pydantic=ActionList)
  task3=Task(description=where+"List only decisions that were stated as decided. Do not list options that were merely discussed.",
             expected_output="A bullet list of decisions.",agent=decision_logger,async_execution=True)
  task4=Task(description="Assemble the summary, the action items as a Markdown table (owner, task, due) and the decisions into one document with the headings Summary, Actions and Decisions. "
                         "Save it with the File Writer tool using the filename notes.md, then return it.",
             expected_output="The meeting notes in Markdown.",agent=formatter,context=[task1,task2,task3])

  #---Step 5: Crew(combine the agents and tasks)
  crew=Crew(agents=[summarizer,action_finder,decision_logger,formatter],tasks=[task1,task2,task3,task4],process=Process.sequential,verbose=False)

  #---Step 6: Run(start the crew and let it work)
  result=crew.kickoff()

  #---Step 7: Result(read the final output)
  found=task2.output.pydantic if task2.output is not None else None
  return MeetingResponse(notes=saved(folder,"notes.md",result.raw),actions=found.items if found else [],trail=crew_trail(crew))


# =====================================================================================
# 4. Data Report Crew - Crew, structured output (output_pydantic) and a pandas tool
# =====================================================================================
class SalesRequest(BaseModel):
  csv_text: str=Field(default=SAMPLES["csv"],min_length=30,max_length=200000,description="CSV with the columns date, region, product, units, unit_price")

class Group(BaseModel):
  name: str
  revenue: float

class SalesStats(BaseModel):
  total_revenue: float=Field(description="Total revenue, units times unit_price")
  total_units: int
  by_region: List[Group]
  by_product: List[Group]
  best_region: str
  worst_region: str
  best_product: str
  first_week_revenue: float
  last_week_revenue: float
  change_percent: float=Field(description="Change from first to last week, in percent")

class CheckRow(BaseModel):
  figure: str
  analyst: float
  recomputed: float
  match: bool

class SalesResponse(BaseModel):
  report: str
  stats: Optional[SalesStats]
  code_check: List[CheckRow]
  all_figures_match: bool
  trail: List[str]

def compute_sales(csv_path):
  df=pd.read_csv(csv_path)
  needed={"date","region","product","units","unit_price"}
  if not needed.issubset(df.columns):
    raise HTTPException(status_code=422,detail="The CSV needs these columns: date, region, product, units, unit_price. Found: "+", ".join(df.columns))
  df["revenue"]=df["units"]*df["unit_price"]
  df["week"]=pd.to_datetime(df["date"]).dt.to_period("W").astype(str)
  weekly=df.groupby("week")["revenue"].sum().round(2)
  first,last=float(weekly.iloc[0]),float(weekly.iloc[-1])
  return {"total_revenue":round(float(df["revenue"].sum()),2),"total_units":int(df["units"].sum()),
          "by_region":{k:round(float(v),2) for k,v in df.groupby("region")["revenue"].sum().items()},
          "by_product":{k:round(float(v),2) for k,v in df.groupby("product")["revenue"].sum().items()},
          "first_week_revenue":first,"last_week_revenue":last,
          "change_percent":round(100*(last-first)/first,1) if first else 0.0}

@app.post("/agent4",response_model=SalesResponse,tags=["Agents"],summary="Data Report Crew")
@guarded
def agent4(req: Optional[SalesRequest]=None):
  """Python computes the figures, a validated model carries them, a reviewer checks them, and code checks again."""
  req=req or SalesRequest()
  folder=tempfile.mkdtemp()
  path=os.path.join(folder,"sales.csv")
  with open(path,"w",encoding="utf-8") as f:
    f.write(req.csv_text)
  try:
    truth=compute_sales(path)
  except HTTPException:
    raise
  except Exception as error:
    raise HTTPException(status_code=422,detail="The CSV could not be read: "+str(error)[:200])

  @tool("Sales Stats")
  def sales_stats(csv_path: str) -> str:
    """Compute exact sales statistics from the CSV file at csv_path. Returns JSON. Always use this tool for numbers."""
    try:
      return json.dumps(compute_sales(csv_path))
    except Exception as error:
      return f"Could not compute: {error}"

  #---Step 3: Agent(define who does the work)
  analyst=Agent(role="Data Analyst",goal="Obtain exact statistics and report them in the required structure",
                backstory="You never calculate by hand. You copy numbers from the tool exactly.",tools=[sales_stats,FileReadTool()],llm=llm,**AGENT_SETTINGS)
  insight_writer=Agent(role="Insight Writer",goal="Write the three most useful findings in plain language",
                       backstory="You explain numbers to a manager. You do not change any number you are given.",llm=llm,**AGENT_SETTINGS)
  reviewer=Agent(role="Reviewer",goal="Check every number in the report against a fresh tool result",
                 backstory="You are strict. A number that differs from the tool is corrected.",tools=[sales_stats,make_writer(folder)],llm=llm,**AGENT_SETTINGS)

  #---Step 4: Task(the first task returns a validated SalesStats object)
  task1=Task(description=f"Call the Sales Stats tool with csv_path={path}. Return the figures in the required structure, copying numbers exactly.",
             expected_output="A SalesStats object.",agent=analyst,output_pydantic=SalesStats)
  task2=Task(description="Using only the statistics provided, write the three most useful findings for a manager, with the supporting numbers.",
             expected_output="Three findings, each with numbers.",agent=insight_writer,context=[task1])
  task3=Task(description=f"Call Sales Stats again with csv_path={path}. Compare every number in the findings with the result and correct any difference. "
                         "Compose a report with the three findings and a table of revenue by region and by product. Save it with the File Writer tool using the filename report.md, then return it.",
             expected_output="The checked report in Markdown.",agent=reviewer,context=[task1,task2])

  #---Step 5: Crew(combine the agents and tasks)
  crew=Crew(agents=[analyst,insight_writer,reviewer],tasks=[task1,task2,task3],process=Process.sequential,verbose=False)

  #---Step 6: Run(start the crew and let it work)
  result=crew.kickoff()

  #---Step 7: Result(read the report, then let code compare the analyst's figures with a fresh calculation)
  stats=task1.output.pydantic if task1.output is not None else None
  rows=[]
  if stats is not None:
    for name,said,real in [("Total revenue",stats.total_revenue,truth["total_revenue"]),("Total units",stats.total_units,truth["total_units"]),
                           ("First-week revenue",stats.first_week_revenue,truth["first_week_revenue"]),("Last-week revenue",stats.last_week_revenue,truth["last_week_revenue"])]:
      rows.append(CheckRow(figure=name,analyst=float(said),recomputed=float(real),match=abs(float(said)-float(real))<0.01))
  return SalesResponse(report=saved(folder,"report.md",result.raw),stats=stats,code_check=rows,all_figures_match=bool(rows) and all(r.match for r in rows),trail=crew_trail(crew))


# =====================================================================================
# 5. Customer Support Flow - Flow, @router, direct LLM calls
# =====================================================================================
class TicketRequest(BaseModel):
  ticket_id: str=Field(default=SAMPLES["tickets"][0]["ticket_id"],min_length=1,max_length=30)
  customer_name: str=Field(default=SAMPLES["tickets"][0]["customer_name"],min_length=2,max_length=80)
  issue_description: str=Field(default=SAMPLES["tickets"][0]["issue_description"],min_length=10,max_length=3000)

class TicketResponse(BaseModel):
  ticket_id: str
  customer_name: str
  category: str
  reason: str
  priority: str
  team: str
  route: str
  resolution: str
  trail: List[str]

class TicketCategory(BaseModel):
  category: Literal["billing","account_access","technical_issue","feature_request","other"]
  reason: str

#---Step 3: State(one shared Pydantic state that every step reads and writes)
class SupportTicketState(BaseModel):
  ticket_id: str=""
  customer_name: str=""
  issue_description: str=""
  category: str=""
  reason: str=""
  priority: str="medium"
  team: str=""
  route: str=""
  resolution: str=""
  satisfaction_score: int=0      #---not used here, stays 0, as in the notes

#---Step 4: Start(@start() is where the flow begins)
class CustomerSupportFlow(Flow[SupportTicketState]):
  @start()
  def receive_ticket(self):
    #---the state is filled from the inputs of kickoff(); this step confirms the ticket
    return f"Ticket {self.state.ticket_id} received from {self.state.customer_name}"

  #---Step 5: Steps(@listen, @router and or_ decide what runs next)
  @listen(receive_ticket)
  def categorize_ticket(self,_):
    prompt=f"""
    Categorize the following customer support ticket into exactly one of these categories:
    billing, account_access, technical_issue, feature_request, other.
    Ticket {self.state.ticket_id} from {self.state.customer_name}:
    {self.state.issue_description}

    Give the category and a one-sentence reason.
    """
    answer=ask(prompt,TicketCategory)
    self.state.category=answer.category
    self.state.reason=answer.reason
    return self.state.category

  @router(categorize_ticket)
  def route_by_category(self,category):
    #---a router can only return a label from this list
    self.state.route=category if category in ("billing","account_access") else "other"
    return self.state.route

  @listen("billing")
  def handle_billing(self):
    self.state.priority="high"
    self.state.team="Billing"
    return "Billing issue handled"

  @listen("account_access")
  def handle_access(self):
    self.state.priority="high"
    self.state.team="Account Security"
    return "Account access issue handled"

  @listen("other")
  def handle_other(self):
    technical=self.state.category=="technical_issue"
    self.state.priority="medium" if technical else "low"
    self.state.team="Technical Support" if technical else "Product"
    return "Other issue handled"

  #---this listens to the three handlers, not to the labels, so it always runs after the priority is set
  @listen(or_(handle_billing,handle_access,handle_other))
  def resolve_ticket(self,_):
    prompt=f"""
    Write a short, courteous reply to this support ticket. Category: {self.state.category}. Priority: {self.state.priority}.
    Do not promise a refund, a date or an outcome. Say what happens next.
    Customer: {self.state.customer_name}
    Ticket: {self.state.issue_description}
    """
    self.state.resolution=ask(prompt)
    return self.state.resolution

@app.post("/agent5",response_model=TicketResponse,tags=["Agents"],summary="Customer Support Flow")
@guarded
def agent5(req: Optional[TicketRequest]=None):
  """One model call sorts the ticket, a router picks the path, and Python sets the priority."""
  req=req or TicketRequest()

  #---Step 6: Run(start the flow)
  support_flow=CustomerSupportFlow()
  support_flow.kickoff(inputs={"ticket_id":req.ticket_id,"customer_name":req.customer_name,"issue_description":req.issue_description})

  #---Step 7: Result(read the final state)
  s=support_flow.state
  return TicketResponse(ticket_id=s.ticket_id,customer_name=s.customer_name,category=s.category,reason=s.reason,priority=s.priority,team=s.team,route=s.route,resolution=s.resolution,trail=flow_trail(support_flow))


# =====================================================================================
# 6. Resume Screening Flow - Flow, @router with a Python threshold, nested Crew
# =====================================================================================
SHORTLIST_SCORE=7

class ResumeIn(BaseModel):
  name: str=Field(min_length=1,max_length=60)
  text: str=Field(min_length=80,max_length=12000)

class ResumeRequest(BaseModel):
  job_description: str=Field(default=SAMPLES["job"],min_length=40,max_length=6000)
  resumes: List[ResumeIn]=Field(default_factory=lambda:[ResumeIn(**r) for r in SAMPLES["resumes"]],min_length=1,max_length=5)

class ResumeResult(BaseModel):
  name: str
  score: int
  decision: Literal["shortlist","reject"]
  reason: str
  output: str=Field(description="Strengths and gaps for a shortlisted candidate, a draft rejection email otherwise")
  trail: List[str]

class ResumeResponse(BaseModel):
  threshold: int
  results: List[ResumeResult]
  results_csv: str=Field(description="The contents of results.csv, saved by the save_results step")

class ScoreCard(BaseModel):
  score: int=Field(description="Fit for the job from 0 to 10, judged only on evidence in the resume")
  reason: str

#---Step 3: State(one shared Pydantic state that every step reads and writes)
class ResumeState(BaseModel):
  name: str=""
  resume: str=""
  job: str=""
  score: int=0
  reason: str=""
  decision: str=""
  output: str=""
  folder: str=""

#---Step 4: Start(@start() is where the flow begins)
class ResumeScreeningFlow(Flow[ResumeState]):
  @start()
  def load_resumes(self):
    #---the route runs this flow once for every resume, so each run loads one
    return f"Resume of {self.state.name} loaded"

  #---Step 5: Steps(a direct call scores, Python decides, a Crew summarises)
  @listen(load_resumes)
  def score_resume(self,_):
    prompt=f"""
    Score this resume from 0 to 10 for the job below. Use only evidence in the resume. Do not assume skills that are not written.
    JOB:
    {self.state.job}

    RESUME:
    {self.state.resume}
    """
    card=ask(prompt,ScoreCard)
    self.state.score=max(0,min(10,card.score))
    self.state.reason=card.reason
    return self.state.score

  @router(score_resume)
  def check_score(self,score):
    #---the rule is plain Python, so the same resume is always treated the same way
    self.state.decision="shortlist" if score>=SHORTLIST_SCORE else "reject"
    return self.state.decision

  @listen("shortlist")
  def summary_crew(self):
    analyst=Agent(role="Resume Summariser",goal="Summarise strengths and gaps of a candidate for a job",
                  backstory="You are fair and specific. You cite evidence from the resume.",llm=llm,**AGENT_SETTINGS)
    task=Task(description="JOB:\n{job}\n\nRESUME:\n{resume}\n\nList the candidate's three strongest points and up to three gaps for this job. Cite the resume.",
              expected_output="Two short bullet lists: Strengths and Gaps.",agent=analyst)
    crew=Crew(agents=[analyst],tasks=[task],process=Process.sequential,verbose=False)
    self.state.output=crew.kickoff(inputs={"job":self.state.job,"resume":self.state.resume}).raw
    return self.state.output

  @listen("reject")
  def draft_rejection(self):
    prompt=f"Write a short, polite email declining the candidate {self.state.name}. Thank them, give no detailed reason, and do not promise to keep their details."
    self.state.output=ask(prompt)
    return self.state.output

  @listen(or_(summary_crew,draft_rejection))
  def save_results(self,_):
    #---one row per resume is added to results.csv
    append_row(self.state.folder,"results.csv",["name","score","decision","reason"],[self.state.name,self.state.score,self.state.decision,self.state.reason])
    return "Row added to results.csv"

@app.post("/agent6",response_model=ResumeResponse,tags=["Agents"],summary="Resume Screening Flow")
@guarded
def agent6(req: Optional[ResumeRequest]=None):
  """Each resume is scored 0 to 10. A score of 7 or more is shortlisted. The others receive a drafted rejection."""
  req=req or ResumeRequest()
  results=[]
  folder=tempfile.mkdtemp()
  for item in req.resumes:

    #---Step 6: Run(start one flow for each resume)
    screening_flow=ResumeScreeningFlow()
    screening_flow.kickoff(inputs={"name":item.name,"resume":item.text,"job":req.job_description,"folder":folder})

    #---Step 7: Result(read the final state)
    s=screening_flow.state
    results.append(ResumeResult(name=item.name,score=s.score,decision=s.decision,reason=s.reason,output=s.output,trail=flow_trail(screening_flow)))
  return ResumeResponse(threshold=SHORTLIST_SCORE,results=results,results_csv=saved(folder,"results.csv",""))


# =====================================================================================
# 7. Bug Triage Flow - Flow, @router to three specialist Crews
# =====================================================================================
class IssueRequest(BaseModel):
  title: str=Field(default=SAMPLES["issues"][0]["title"],min_length=3,max_length=200)
  body: str=Field(default=SAMPLES["issues"][0]["body"],min_length=10,max_length=5000)
  docs_url: Optional[HttpUrl]=Field(default=None,description="Documentation page used to answer questions")

class IssueResponse(BaseModel):
  type: Literal["bug","feature","question"]
  severity: Optional[str]
  labels: List[str]
  comment: str
  trail: List[str]

class IssueType(BaseModel):
  type: Literal["bug","feature","question"]
  reason: str

class Severity(BaseModel):
  severity: Literal["critical","high","medium","low"]
  reason: str

#---Step 3: State(one shared Pydantic state that every step reads and writes)
class IssueState(BaseModel):
  title: str=""
  body: str=""
  docs_url: str=""
  type: str=""
  severity: str=""
  result: str=""
  labels: List[str]=[]

#---Step 4: Start(@start() is where the flow begins)
class BugTriageFlow(Flow[IssueState]):
  @start()
  def read_issue(self):
    return "Issue read"

  #---Step 5: Steps(a router hands the issue to one of three Crews)
  @listen(read_issue)
  def classify_issue(self,_):
    prompt=f"""
    Classify this issue as exactly one of: bug (something is broken), feature (a request for new behaviour) or question (the person asks how to do something).
    Title: {self.state.title}
    Body: {self.state.body}
    """
    answer=ask(prompt,IssueType)
    self.state.type=answer.type
    return self.state.type

  @router(classify_issue)
  def route_type(self,issue_type):
    return issue_type

  @listen("bug")
  def bug_crew(self):
    rater=Agent(role="Severity Analyst",goal="Rate the severity of a bug",
                backstory="You rate by user impact: critical means data loss or security, high means a main feature is unusable.",llm=llm,**AGENT_SETTINGS)
    writer=Agent(role="Reproduction Writer",goal="Write clear reproduction steps",
                 backstory="You write numbered steps that another person can follow exactly.",llm=llm,**AGENT_SETTINGS)
    task1=Task(description="Issue: {title}\n{body}\n\nRate the severity as critical, high, medium or low, with one sentence of reasoning.",
               expected_output="A severity and a reason.",agent=rater,output_pydantic=Severity)
    task2=Task(description="Issue: {title}\n{body}\n\nWrite numbered reproduction steps, the expected result and the actual result. Use only what the report states, and mark anything missing as 'needs information'.",
               expected_output="Numbered steps with expected and actual results.",agent=writer,context=[task1])
    crew=Crew(agents=[rater,writer],tasks=[task1,task2],process=Process.sequential,verbose=False)
    result=crew.kickoff(inputs={"title":self.state.title,"body":self.state.body})
    rated=task1.output.pydantic if task1.output is not None else None
    self.state.severity=rated.severity if rated else "unrated"
    self.state.result=f"**Severity: {self.state.severity}**"+(f" - {rated.reason}" if rated else "")+"\n\n"+result.raw
    return self.state.result

  @listen("feature")
  def feature_crew(self):
    analyst=Agent(role="Product Analyst",goal="Turn a request into a user story with acceptance criteria",
                  backstory="You write for engineers: one story, three to five testable criteria.",llm=llm,**AGENT_SETTINGS)
    task=Task(description="Request: {title}\n{body}\n\nWrite one user story in the form 'As a ..., I want ..., so that ...' and three to five acceptance criteria.",
              expected_output="A user story and acceptance criteria.",agent=analyst)
    crew=Crew(agents=[analyst],tasks=[task],process=Process.sequential,verbose=False)
    self.state.result=crew.kickoff(inputs={"title":self.state.title,"body":self.state.body}).raw
    return self.state.result

  @listen("question")
  def answer_crew(self):
    has_docs=self.state.docs_url.startswith("http")
    helper=Agent(role="Support Answerer",goal="Answer a question using only the documentation page provided",
                 backstory="You answer from the documentation. If the documentation does not contain the answer, you say so.",
                 tools=[ScrapeWebsiteTool()] if has_docs else [],llm=llm,**AGENT_SETTINGS)
    where=f"Read the documentation at {self.state.docs_url} with ScrapeWebsiteTool. " if has_docs else "No documentation page was supplied, so say that you cannot confirm the answer. "
    task=Task(description=where+"Question: {title}\n{body}\n\nAnswer in a short reply. Quote the part of the documentation you used.",
              expected_output="A short reply that quotes the documentation.",agent=helper)
    crew=Crew(agents=[helper],tasks=[task],process=Process.sequential,verbose=False)
    self.state.result=crew.kickoff(inputs={"title":self.state.title,"body":self.state.body}).raw
    return self.state.result

  @listen(or_(bug_crew,feature_crew,answer_crew))
  def post_reply(self,_):
    #---the labels are built by code from the type (and the severity of a bug); nothing is posted anywhere
    self.state.labels=[f"type:{self.state.type}"]+([f"severity:{self.state.severity}"] if self.state.type=="bug" else [])
    return self.state.labels

@app.post("/agent7",response_model=IssueResponse,tags=["Agents"],summary="Bug Triage Flow")
@guarded
def agent7(req: Optional[IssueRequest]=None):
  """The issue is classified, handled by a bug, feature or answer Crew, and labelled by code. Nothing is posted anywhere."""
  req=req or IssueRequest()

  #---Step 6: Run(start the flow)
  triage_flow=BugTriageFlow()
  triage_flow.kickoff(inputs={"title":req.title,"body":req.body,"docs_url":str(req.docs_url or "")})

  #---Step 7: Result(read the final state; labels are built by code)
  s=triage_flow.state
  return IssueResponse(type=s.type,severity=s.severity or None,labels=s.labels,comment=s.result,trail=flow_trail(triage_flow))


# =====================================================================================
# 8. Content Repurposing Flow - Flow, three Crews in parallel, joined with and_
# =====================================================================================
class ArticleRequest(BaseModel):
  article: str=Field(default=SAMPLES["article"],min_length=300,max_length=20000,description="The article text. Used when url is empty.")
  url: Optional[HttpUrl]=Field(default=None,description="Optional. If given, the page is read with ScrapeWebsiteTool instead of using the article text.")

class PackChecks(BaseModel):
  tweets: int
  longest_tweet: int
  every_tweet_within_280: bool
  linkedin_words: int
  linkedin_near_150_words: bool

class ArticleResponse(BaseModel):
  files_saved: List[str]
  linkedin: str
  thread: List[str]
  newsletter: str
  checks: PackChecks
  trail: List[str]

class KeyPoints(BaseModel):
  title: str
  audience: str
  tone: str
  points: List[str]=Field(description="Five to seven key points, each a complete sentence")

#---Step 3: State(one shared Pydantic state that every step reads and writes)
class PackState(BaseModel):
  article: str=""
  title: str=""
  points: str=""
  linkedin: str=""
  thread: str=""
  newsletter: str=""
  url: str=""
  folder: str=""
  files: List[str]=[]

#---Step 4: Start(@start() is where the flow begins)
class ContentRepurposingFlow(Flow[PackState]):
  @start()
  def load_article(self):
    #---URL or file: a web address is read with ScrapeWebsiteTool, otherwise the pasted text is used
    if self.state.url:
      try:
        page=str(ScrapeWebsiteTool(website_url=self.state.url).run())
      except Exception as error:
        raise HTTPException(status_code=422,detail="The page could not be read: "+str(error)[:150])
      if len(page.strip())<300:
        raise HTTPException(status_code=422,detail="The page returned too little text to repurpose. Paste the article text instead.")
      self.state.article=page[:20000]
    return "Article loaded"

  #---Step 5: Steps(three listeners on one step run side by side, and_ waits for all of them)
  @listen(load_article)
  def extract_points(self,_):
    prompt="Read the article and return its title, intended audience, tone and five to seven key points. Use only the article.\n\nARTICLE:\n"+self.state.article
    answer=ask(prompt,KeyPoints)
    self.state.title=answer.title
    self.state.points="\n".join(f"- {p}" for p in answer.points)
    return self.state.points

  @listen(extract_points)
  def linkedin_crew(self,_):
    self.state.linkedin=one_writer_crew("LinkedIn Writer","Write a LinkedIn post that stays faithful to the key points",
      "Article title: {title}\nKey points:\n{points}\n\nWrite a LinkedIn post: one opening hook line, then about 150 words. No hashtags. Use only the key points.",
      "A LinkedIn post of about 150 words.",self.state)
    return self.state.linkedin

  @listen(extract_points)
  def twitter_crew(self,_):
    self.state.thread=one_writer_crew("Thread Writer","Write a six-tweet thread that stays faithful to the key points",
      "Article title: {title}\nKey points:\n{points}\n\nWrite a thread of exactly six tweets. Number them 1/6 to 6/6, one per line. Each must be 280 characters or fewer.",
      "Six numbered tweets, one per line.",self.state)
    return self.state.thread

  @listen(extract_points)
  def newsletter_crew(self,_):
    self.state.newsletter=one_writer_crew("Newsletter Writer","Write a newsletter section that stays faithful to the key points",
      "Article title: {title}\nKey points:\n{points}\n\nWrite a two-sentence introduction and exactly three bullet points.",
      "An introduction and three bullets.",self.state)
    return self.state.newsletter

  @listen(and_(linkedin_crew,twitter_crew,newsletter_crew))
  def publish_pack(self):
    #---three files are saved, one for each Crew
    write_file(self.state.folder,"linkedin_post.md",self.state.linkedin)
    write_file(self.state.folder,"twitter_thread.md",self.state.thread)
    write_file(self.state.folder,"newsletter.md",self.state.newsletter)
    self.state.files=["linkedin_post.md","twitter_thread.md","newsletter.md"]
    return self.state.files

#---each of the three Crews below is built by this helper: one Agent, one Task, one Crew
def one_writer_crew(role,goal,description,expected,state):
  writer=Agent(role=role,goal=goal,backstory="You write from the given key points only. You do not add facts.",llm=llm,**AGENT_SETTINGS)
  task=Task(description=description,expected_output=expected,agent=writer)
  crew=Crew(agents=[writer],tasks=[task],process=Process.sequential,verbose=False)
  return crew.kickoff(inputs={"title":state.title,"points":state.points}).raw

@app.post("/agent8",response_model=ArticleResponse,tags=["Agents"],summary="Content Repurposing Flow")
@guarded
def agent8(req: Optional[ArticleRequest]=None):
  """One article becomes a LinkedIn post, a six-tweet thread and a newsletter. Length rules are checked by code."""
  req=req or ArticleRequest()

  #---Step 6: Run(start the flow)
  pack_flow=ContentRepurposingFlow()
  pack_flow.kickoff(inputs={"article":req.article,"url":str(req.url or ""),"folder":tempfile.mkdtemp()})

  #---Step 7: Result(read the final state; code checks the lengths)
  s=pack_flow.state
  tweets=[t.strip() for t in s.thread.splitlines() if t.strip()]
  words=len(s.linkedin.split())
  checks=PackChecks(tweets=len(tweets),longest_tweet=max([len(t) for t in tweets] or [0]),every_tweet_within_280=all(len(t)<=280 for t in tweets),
                    linkedin_words=words,linkedin_near_150_words=120<=words<=190)
  return ArticleResponse(files_saved=s.files,linkedin=s.linkedin,thread=tweets,newsletter=s.newsletter,checks=checks,trail=flow_trail(pack_flow))


# =====================================================================================
# 9. Lead Qualification Flow - Flow, enrichment Crew, @router, outreach Crew
# =====================================================================================
class LeadRequest(BaseModel):
  name: str=Field(default=SAMPLES["leads"][0]["name"],min_length=2,max_length=100)
  website: HttpUrl=Field(default=HttpUrl(SAMPLES["leads"][0]["website"]))
  offer: str=Field(default=SAMPLES["leads"][0]["offer"],min_length=5,max_length=300,description="What you sell, in one sentence")

class LeadResponse(BaseModel):
  leads_csv: str=Field(description="The contents of leads.csv, written by the log_lead step")
  rating: Literal["hot","warm","cold"]
  points: int
  reasons: str
  action: str
  profile: str
  email: Optional[str]
  trail: List[str]

class CompanyProfile(BaseModel):
  readable: bool=Field(description="True only if the website text describes the company")
  what_they_do: str
  size_signals: List[str]
  buying_signals: List[str]

class LeadScore(BaseModel):
  rating: Literal["hot","warm","cold"]
  points: int=Field(description="Fit with the offer, 0 to 100")
  reasons: List[str]

#---Step 3: State(one shared Pydantic state that every step reads and writes)
class LeadState(BaseModel):
  name: str=""
  website: str=""
  offer: str=""
  profile: str=""
  readable: bool=False
  rating: str=""
  points: int=0
  reasons: str=""
  email: str=""
  action: str=""
  folder: str=""

#---Step 4: Start(@start() is where the flow begins)
class LeadQualificationFlow(Flow[LeadState]):
  @start()
  def capture_lead(self):
    return "Lead captured"

  #---Step 5: Steps(a Crew reads the site, a router rates the lead, only a hot lead gets an email)
  @listen(capture_lead)
  def enrich_crew(self,_):
    researcher=Agent(role="Company Researcher",goal="Describe a company from its website",
                     backstory="You report only what the website text says.",tools=[ScrapeWebsiteTool()],llm=llm,**AGENT_SETTINGS)
    task=Task(description="Read {website} with ScrapeWebsiteTool. Describe what the company does, any signals of its size, and any signals that it might need: {offer}. If the page returns little text, set readable to false.",
              expected_output="A CompanyProfile.",agent=researcher,output_pydantic=CompanyProfile)
    crew=Crew(agents=[researcher],tasks=[task],process=Process.sequential,verbose=False)
    crew.kickoff(inputs={"website":self.state.website,"offer":self.state.offer})
    found=task.output.pydantic if task.output is not None else None
    self.state.readable=bool(found and found.readable)
    self.state.profile=(f"{found.what_they_do}\nSize: {'; '.join(found.size_signals)}\nSignals: {'; '.join(found.buying_signals)}" if found else "The profile could not be produced.")
    return self.state.profile

  @router(enrich_crew)
  def score_lead(self,_):
    if not self.state.readable:
      #---no readable website, no evidence: code decides, not the model
      self.state.rating,self.state.points,self.state.reasons="cold",0,"The website could not be read, so there is no evidence of fit."
    else:
      prompt=f"""
      We sell: {self.state.offer}

      Company profile:
      {self.state.profile}

      Rate this lead hot (clear need and a fit now), warm (possible fit, not urgent) or cold (no evidence of fit).
      Give a score from 0 to 100 and two or three reasons, using only the profile.
      """
      answer=ask(prompt,LeadScore)
      self.state.rating,self.state.points,self.state.reasons=answer.rating,answer.points,"; ".join(answer.reasons)
    return self.state.rating

  @listen("hot")
  def outreach_crew(self):
    writer=Agent(role="Outreach Writer",goal="Write a short, specific first email",
                 backstory="You refer to something real from the company profile and make no claims about our product beyond the offer.",llm=llm,**AGENT_SETTINGS)
    task=Task(description="We sell: {offer}\nCompany: {name}\nProfile:\n{profile}\n\nWrite a first email of at most 120 words with a subject line. Refer to one specific detail from the profile.",
              expected_output="An email with a subject line.",agent=writer)
    crew=Crew(agents=[writer],tasks=[task],process=Process.sequential,verbose=False)
    self.state.email=crew.kickoff(inputs={"offer":self.state.offer,"name":self.state.name,"profile":self.state.profile}).raw
    self.state.action="Email drafted"
    return self.state.email

  @listen("warm")
  def add_to_nurture(self):
    self.state.action="Added to the weekly tips list"
    return self.state.action

  @listen("cold")
  def archive_lead(self):
    self.state.action="Archived, no action"
    return self.state.action

  @listen(or_(outreach_crew,add_to_nurture,archive_lead))
  def log_lead(self,_):
    #---the lead is added to leads.csv
    append_row(self.state.folder,"leads.csv",["name","website","rating","points","action"],[self.state.name,self.state.website,self.state.rating,self.state.points,self.state.action])
    return "Row added to leads.csv"

@app.post("/agent9",response_model=LeadResponse,tags=["Agents"],summary="Lead Qualification Flow")
@guarded
def agent9(req: Optional[LeadRequest]=None):
  """The website is read, the lead is rated hot, warm or cold, and only hot leads receive a drafted email. Nothing is sent."""
  req=req or LeadRequest()

  #---Step 6: Run(start the flow)
  lead_flow=LeadQualificationFlow()
  folder=tempfile.mkdtemp()
  lead_flow.kickoff(inputs={"name":req.name,"website":str(req.website),"offer":req.offer,"folder":folder})

  #---Step 7: Result(read the final state)
  s=lead_flow.state
  return LeadResponse(leads_csv=saved(folder,"leads.csv",""),rating=s.rating,points=s.points,reasons=s.reasons,action=s.action,profile=s.profile,email=s.email or None,trail=flow_trail(lead_flow))


# =====================================================================================
# 10. Approval Report Flow - Flow with a human decision between two calls
# =====================================================================================
class ReportRequest(BaseModel):
  topic: str=Field(default=SAMPLES["report"]["topic"],min_length=3,max_length=150)
  audience: str=Field(default=SAMPLES["report"]["audience"],min_length=3,max_length=150)
  draft: Optional[str]=Field(default=None,max_length=30000,description="Leave empty to start. Send the draft back with a decision to continue.")
  decision: Optional[Literal["approved","revise"]]=Field(default=None,description="Your decision on the draft")
  feedback: str=Field(default="",max_length=3000,description="Required when the decision is 'revise'")

class ReportResponse(BaseModel):
  status: Literal["awaiting_review","published"]
  draft: str
  final: Optional[str]=Field(description="Set only after the draft is approved")
  saved_as: Optional[str]=Field(default=None,description="final.md, written by publish_report")
  next_step: str
  trail: List[str]

class Finding(BaseModel):
  claim: str
  source: str=Field(description="Web address of the source")

class ResearchPack(BaseModel):
  findings: List[Finding]

#---Step 3: State(one shared Pydantic state that every step reads and writes)
class ApprovalReportState(BaseModel):
  topic: str=""
  audience: str=""
  draft: str=""
  decision: str=""
  feedback: str=""
  findings: str=""
  status: str=""
  final: str=""
  folder: str=""

#---a web request cannot wait for a person, so this provider pauses the Flow instead of asking on a console
class PauseForPerson(HumanFeedbackProvider):
  def request_feedback(self,context,flow):
    raise HumanFeedbackPending(context=context)

#---Step 4: Start(@start() is where the flow begins)
class ApprovalReportFlow(Flow[ApprovalReportState]):
  @start()
  def collect_topic(self):
    return "Topic collected"

  #---Step 5: Steps(a new report starts the Crews; a draft that comes back with a decision goes straight to that decision)
  @router(collect_topic)
  def check_decision(self,_):
    return self.state.decision or "new"

  @listen("new")
  def research_crew(self):
    researcher=Agent(role="Researcher",goal="Collect sourced claims about the topic",backstory="You record each claim with the address of its source.",tools=[duckduckgo_search],llm=llm,**AGENT_SETTINGS)
    analyst=Agent(role="Analyst",goal="Select the findings most useful to the audience",backstory="You prefer findings that are specific and recent.",llm=llm,**AGENT_SETTINGS)
    checker=Agent(role="Fact Checker",goal="Remove every claim that has no source",backstory="You are strict: a claim without a source address is deleted.",llm=llm,**AGENT_SETTINGS)
    task1=Task(description="Search the web about: {topic}. List 8 to 10 claims, each with its source address.",expected_output="Claims with source addresses.",agent=researcher)
    task2=Task(description="Audience: {audience}. From the claims, keep the 6 to 8 most useful to this audience.",expected_output="A shorter list of claims.",agent=analyst,context=[task1])
    task3=Task(description="Remove any claim that has no source address. Return the remaining claims with their sources.",expected_output="Checked findings.",agent=checker,context=[task2],output_pydantic=ResearchPack)
    crew=Crew(agents=[researcher,analyst,checker],tasks=[task1,task2,task3],process=Process.sequential,verbose=False)
    result=crew.kickoff(inputs={"topic":self.state.topic,"audience":self.state.audience})
    pack=task3.output.pydantic if task3.output is not None else None
    self.state.findings="\n".join(f"- {f.claim} (source: {f.source})" for f in pack.findings) if pack else result.raw
    return self.state.findings

  @listen(research_crew)
  def writing_crew(self,_):
    writer=Agent(role="Report Writer",goal="Write a clear report for the audience from checked findings only",backstory="You cite the source address of every claim and add nothing else.",llm=llm,**AGENT_SETTINGS)
    task=Task(description="Write a report on {topic} for {audience}, about 500 words, with a title, short sections and a Sources list. Use only these findings:\n{findings}",
              expected_output="A report in Markdown.",agent=writer)
    crew=Crew(agents=[writer],tasks=[task],process=Process.sequential,verbose=False)
    self.state.draft=crew.kickoff(inputs={"topic":self.state.topic,"audience":self.state.audience,"findings":self.state.findings}).raw
    return self.state.draft

  @listen("approved")
  def publish_report(self):
    self.state.final=self.state.draft
    write_file(self.state.folder,"final.md",self.state.final)
    self.state.status="published"
    return "Saved final.md"

  @listen("revise")
  def revision_crew(self):
    editor=Agent(role="Revision Editor",goal="Apply the reviewer's comments to the draft and change nothing else",backstory="You make only the changes that were requested.",llm=llm,**AGENT_SETTINGS)
    task=Task(description="Reviewer's comments:\n{feedback}\n\nDraft:\n{draft}\n\nApply the comments and return the complete revised report. Change nothing that was not requested.",
              expected_output="The full revised report in Markdown.",agent=editor)
    crew=Crew(agents=[editor],tasks=[task],process=Process.sequential,verbose=False)
    self.state.draft=crew.kickoff(inputs={"feedback":self.state.feedback,"draft":self.state.draft}).raw
    return self.state.draft

  @listen(or_(writing_crew,revision_crew))
  @human_feedback(message="Read the draft. Approve it, or ask for changes.",provider=PauseForPerson())
  def review_draft(self,_):
    #---@human_feedback: the flow stops after this step and waits for a person; the decision arrives with the next call
    self.state.status="awaiting_review"
    return self.state.draft

@app.post("/agent10",response_model=ReportResponse,tags=["Agents"],summary="Approval Report Flow")
@guarded
def agent10(req: Optional[ReportRequest]=None):
  """Call 1: topic and audience return a draft. Call 2: send the draft back with decision 'approved' (publish) or 'revise' plus feedback."""
  req=req or ReportRequest()
  if req.draft and not req.decision:
    raise HTTPException(status_code=422,detail="A draft was sent without a decision. Send decision 'approved' or 'revise'.")
  if req.decision and not req.draft:
    raise HTTPException(status_code=422,detail="A decision was sent without the draft it applies to.")
  if req.decision=="revise" and not req.feedback.strip():
    raise HTTPException(status_code=422,detail="Feedback is required when the decision is 'revise'.")

  #---Step 6: Run(start the flow; each call runs it once)
  report_flow=ApprovalReportFlow()
  paused=isinstance(report_flow.kickoff(inputs={"topic":req.topic,"audience":req.audience,"draft":req.draft or "","decision":req.decision or "","feedback":req.feedback,"folder":tempfile.mkdtemp()}),HumanFeedbackPending)

  #---Step 7: Result(read the final state)
  s=report_flow.state
  nxt="Published. Nothing else to do." if s.status=="published" else "Read the draft, then call /agent10 again with the same topic and audience, this draft, and decision 'approved' or 'revise' (with feedback)."
  return ReportResponse(status=s.status,draft=s.draft,final=s.final or None,saved_as="final.md" if s.final else None,next_step=nxt,trail=flow_trail(report_flow)+(["review_draft (waiting for a person)"] if paused else []))


# =====================================================================================
# END OF THE TEN AGENTS
# =====================================================================================


#---GET routes: opening /agent1 ... /agent10 in a browser describes the agent. Add ?run=true to run it on its sample input.
AGENTS={1:(agent1,BlogRequest,"Blog Post Crew"),2:(agent2,CompetitorRequest,"Competitor Analysis Crew"),3:(agent3,MeetingRequest,"Meeting Notes Crew"),
        4:(agent4,SalesRequest,"Data Report Crew"),5:(agent5,TicketRequest,"Customer Support Flow"),6:(agent6,ResumeRequest,"Resume Screening Flow"),
        7:(agent7,IssueRequest,"Bug Triage Flow"),8:(agent8,ArticleRequest,"Content Repurposing Flow"),9:(agent9,LeadRequest,"Lead Qualification Flow"),
        10:(agent10,ReportRequest,"Approval Report Flow")}

def add_get_route(n,function,Model,title):
  @app.get(f"/agent{n}",tags=["Agents"],summary=title+" (GET: description, or run with ?run=true)",response_model=None,name=f"agent{n}_get")
  def agent_get(request: Request,run: bool=False):
    #---query parameters that match a text field of the input override its sample value
    params={k:v for k,v in request.query_params.items() if k in Model.model_fields and k!="run"}
    if run or params:
      try:
        req=Model(**params)
      except Exception as error:
        raise HTTPException(status_code=422,detail=str(error)[:300])
      return function(req)
    base=str(request.base_url).rstrip("/")
    example=Model().model_dump(mode="json")
    return {"agent":f"/agent{n}","name":title,"description":(function.__doc__ or "").strip(),
            "how_to_call":{"POST":f"{base}/agent{n}","body":"JSON that matches input_schema. An empty body or {} runs the example input.","run_the_example_in_a_browser":f"{base}/agent{n}?run=true"},
            "example_input":example,"input_schema":Model.model_json_schema()}
  return agent_get

for _n,(_function,_Model,_title) in AGENTS.items():
  add_get_route(_n,_function,_Model,_title)

@app.get("/health",tags=["Status"])
def health():
  return {"ok":True,"has_key":bool(GEMINI_API_KEY),"model":GEMINI_MODEL,"routes":[f"/agent{i}" for i in range(1,11)]}


# =====================================================================================
# THE WEBSITE, served at "/". Everything on it is generated from the code above.
# =====================================================================================
WEIGHTS=[("agents","Agents",2,"Each agent has a role, a goal and tools to configure."),
         ("tasks","Tasks",1,"Each task needs a description, an expected output and context."),
         ("tools","Distinct tools",1,"Each tool is another system that can fail."),
         ("steps","Flow steps",1,"Each @start or @listen method is a step to keep consistent."),
         ("routers","Routers",3,"A router creates branches, so some paths may not run."),
         ("parallel","Extra parallel branches",2,"Concurrent work must be joined, and shares the rate limit."),
         ("crews","Crews inside a Flow",2,"A nested Crew is a second layer of orchestration."),
         ("manager","Manager agent",4,"A manager decides the delegation, so the order is not fixed."),
         ("human","Human gate",4,"A person must be waited for, and the run must be resumable."),
         ("loop","Loops",3,"A loop has no fixed length."),
         ("models","Structured models",1,"Each Pydantic model is a contract to define and validate.")]
SAFEGUARDS=[("structured","Structured output",20,"The answer is validated against a Pydantic model, not read as free text."),
            ("deterministic","Code does the exact work",20,"Counting, thresholds, saving and joining are done by Python, not by the model."),
            ("grounding","Grounded in a source",15,"A tool brings the facts in: a file, a page or a search."),
            ("verification","Independent check",15,"A separate agent or a code check re-examines the result."),
            ("routing","Constrained routing",10,"A router can return only labels from a fixed list."),
            ("human","Human approval",20,"A person approves the result before it is published.")]
QUADRANTS={(False,False):"Simple Crews",(True,False):"Complex Crews",(False,True):"Flows with direct LLM calls",(True,True):"Flows orchestrating Crews"}
STEPS_CREW=[("Setup","Import crewai and set the API key."),("LLM","Choose the model and its settings."),("Agent","Define who does the work."),("Task","Define what needs to be done."),
            ("Crew","Combine the agents and tasks."),("Run","Start the crew and let it work."),("Result","Read the final output.")]
STEPS_FLOW=[("Setup","Import crewai and set the API key."),("LLM","Choose the model and its settings."),("State","One shared Pydantic state that every step reads and writes."),
            ("Start","@start() is where the flow begins."),("Steps","@listen, @router, or_ and and_ decide what runs next. Crews run inside steps."),
            ("Run","flow.kickoff() starts the flow."),("Result","Read the final state or the last step's output.")]

PROJECTS=[
 {"n":1,"title":"Blog Post Crew","kind":"Crew","tags":["Crew","Sequential"],
  "problem":"Writing a well-researched blog post takes hours of searching, drafting and editing for every single topic.",
  "features":{"agents":3,"tasks":3,"tools":2},
  "controls":{"grounding":"DuckDuckGo search supplies the facts the Writer may use.","verification":"The Editor checks each claim against the research notes."},
  "why":"A blog post is judged by a reader, not checked against one correct answer, so three agents in a fixed order are enough: complexity 3. Precision is 3 because the only safeguards are a search tool that supplies the facts and an Editor who removes unsupported claims. A wrong sentence costs a correction, not a loss, so the heavier controls of the upper half would add effort without adding value. This is the right place for an open-ended writing task.",
  "steps":{"3":"Three agents: a Researcher with the DuckDuckGo tool, a Writer with no tools and an Editor with a file writer.","4":"Three tasks. The Writer's task receives the research as context, and the Editor's task receives both.",
           "5":"Crew(...process=Process.sequential) runs the three tasks in order, each feeding the next.","6":"kickoff(inputs={\"topic\": ...}) fills {topic} in every task description.","7":"result.raw is the final text. The route returns the saved blog_post.md."}},
 {"n":2,"title":"Competitor Analysis Crew","kind":"Crew","tags":["Crew","Hierarchical"],
  "problem":"Tracking what competitors ship, charge and say takes hours of manual browsing every week.",
  "features":{"agents":4,"tasks":4,"tools":3,"manager":1},
  "controls":{"grounding":"ScrapeWebsiteTool reads the competitor pages and DuckDuckGo supplies the news.","verification":"The Pricing Analyst re-reads the scraped text and writes 'not found' where it holds no price."},
  "why":"A manager model decides who does what, which is the only place in this set where a model, not the programmer, sets the order of work. That freedom lifts complexity to 5. Precision stays at 3: competitor pages are read as free text, with no structured output and no code check. The region suits open-ended research, and its weakness is stated in the design, so the agents write 'not found' instead of guessing a price.",
  "steps":{"3":"Four specialists: Site Scraper, Pricing Analyst, News Analyst and Strategist. None of them is the manager.","4":"Four tasks with no agent attached. In a hierarchical crew the manager chooses who does each one.",
           "5":"process=Process.hierarchical with manager_llm=llm adds a manager that delegates and checks the work. The notes draw the three specialists as working in parallel. In CrewAI the manager hands out one task at a time: marking the tasks async did not make them overlap in a test, so the project does not claim parallel work and earns no points for it.","6":"kickoff receives the competitor addresses and your product.","7":"The SWOT report is saved as report.md and returned."}},
 {"n":3,"title":"Meeting Notes Crew","kind":"Crew","tags":["Crew","Parallel tasks"],
  "problem":"After every meeting someone must write the summary and chase action items, and details get lost.",
  "features":{"agents":4,"tasks":4,"tools":2,"parallel":2,"models":1},
  "controls":{"structured":"The action items are validated against ActionList (owner, task, due).","grounding":"FileReadTool reads the transcript, and every agent works from that one file."},
  "why":"Three agents read the same transcript at the same time and a fourth joins their results: complexity 5. Precision is 4: the action items are validated against a model and every agent reads the same file, but nothing checks that the summary is faithful. That is acceptable because a person reads meeting notes the same day.",
  "steps":{"3":"Four agents. Three read the transcript at the same time, so each needs its own agent. A Formatter joins their work.","4":"Three tasks run with async_execution=True. The last task waits for all three through context. The action items use output_pydantic.",
           "5":"A sequential crew still runs the three asynchronous tasks side by side.","6":"kickoff runs once. No inputs are needed because the transcript is a file.","7":"notes.md is returned together with the validated action items."}},
 {"n":4,"title":"Data Report Crew","kind":"Crew","tags":["Crew","Structured output"],
  "problem":"Weekly sales CSV files pile up, and turning them into a readable report is slow, repetitive work.",
  "features":{"agents":3,"tasks":3,"tools":3,"models":1},
  "controls":{"structured":"SalesStats is returned through output_pydantic.","deterministic":"pandas calculates every figure inside the Sales Stats tool.","grounding":"The CSV file is read by a tool, not pasted into a prompt.","verification":"The Reviewer re-runs the tool, and code compares four totals again."},
  "why":"Few moving parts (complexity 3) and high precision (7): pandas calculates every number, a validated model carries the figures, a reviewer compares them with a fresh calculation, and code compares once more. The graph recommends a Flow for this region. The notes list this project as a Crew, so we keep a Crew and let code, not agents, supply the precision.",
  "steps":{"3":"A Data Analyst that must use the Sales Stats tool, an Insight Writer, and a Reviewer who re-runs the tool.","4":"Task 1 returns a SalesStats object through output_pydantic, so the figures are a validated structure.",
           "5":"A sequential crew: analyse, write, review.","6":"kickoff runs the crew. Afterwards Python recomputes four totals.","7":"The report, the validated statistics and a table of code checks are returned."}},
 {"n":5,"title":"Customer Support Flow","kind":"Flow","tags":["Flow","Router"],
  "problem":"Support tickets arrive mixed together, and each one has to be read, sorted and sent to the right person.",
  "features":{"steps":6,"routers":1,"models":1},
  "controls":{"structured":"TicketCategory limits the category to five values.","deterministic":"Python sets the priority and the team in the handlers.","routing":"The router returns only billing, account_access or other."},
  "why":"The smallest design in the set: complexity 3. Precision 5 comes from three safeguards: the category is validated, Python sets the priority and team, and the router can return only fixed labels. This is the notes' own example of a Flow with direct LLM calls and no Crew.",
  "steps":{"3":"SupportTicketState holds ticket_id, customer_name, issue_description, category, priority, team, resolution and a satisfaction_score that is not used and stays 0.","4":"@start() receive_ticket: the state is filled from the ticket that arrives.",
           "5":"categorize_ticket makes one direct LLM call. @router route_by_category sends the ticket to billing, account_access or other. handle_billing and handle_access set the priority to high, handle_other covers technical and feature requests, and or_ starts resolve_ticket.","6":"kickoff(inputs={ticket_id, customer_name, issue_description}) fills the state and runs the flow.","7":"The final state is read: category, priority, team and resolution."}},
 {"n":6,"title":"Resume Screening Flow","kind":"Flow","tags":["Flow","Router"],
  "problem":"One job post brings hundreds of resumes, and reading each one takes recruiters days.",
  "features":{"steps":4,"routers":1,"crews":1,"agents":1,"tasks":1,"tools":1,"models":1},
  "controls":{"structured":"ScoreCard (score, reason) is validated.","deterministic":"score >= 7 is plain Python, not a model's opinion.","grounding":"The score may use only the resume text supplied.","routing":"The router returns only shortlist or reject."},
  "why":"The model makes one judgement, a score from 0 to 10. The decision is a number in Python: 7 or more is shortlisted, so the same resume always receives the same decision. A Crew runs only for shortlisted candidates, which keeps complexity at 4 while precision reaches 7.",
  "steps":{"3":"ResumeState holds the resume, the job, the score and the decision.","4":"@start() load_resumes reads the resume (the route starts one flow for each resume).",
           "5":"score_resume is a direct LLM call. The router applies score >= 7 in Python. A shortlisted resume goes to a summary Crew, the others to a drafted rejection, and or_ runs save_results, which adds a row to results.csv.","6":"One flow is started for each resume.","7":"A list of results: name, score, decision, reason and output."}},
 {"n":7,"title":"Bug Triage Flow","kind":"Flow","tags":["Flow + Crews","Router"],
  "problem":"Issues pile up in the tracker without labels, so urgent bugs sit unseen next to feature requests.",
  "features":{"steps":3,"routers":1,"crews":3,"agents":4,"tasks":4,"tools":1,"models":2},
  "controls":{"structured":"IssueType and Severity are validated.","deterministic":"Labels are built by code from the type and the severity.","grounding":"The question Crew quotes the documentation page it reads.","routing":"The router returns only bug, feature or question."},
  "why":"Three specialist Crews sit behind one router, so each issue takes exactly one path. Each Crew adds agents and tasks, which raises complexity to 7. The router returns only bug, feature or question, labels come from code, and the question Crew answers from a documentation page. No independent check is included, so precision stops at 7.",
  "steps":{"3":"IssueState holds the issue text, its type, its severity and the result.","4":"@start() read_issue.",
           "5":"classify_issue, then @router. Each label starts a Crew: bug (severity and steps), feature (user story) or question (answer from the docs). or_ runs post_reply, which builds the labels. Nothing is posted anywhere.","6":"kickoff(inputs={title, body, docs_url}).","7":"The type, severity, labels and the comment to post."}},
 {"n":8,"title":"Content Repurposing Flow","kind":"Flow","tags":["Flow + Crews","and_"],
  "problem":"One long article should become a LinkedIn post, a tweet thread and a newsletter, but rewriting it three times wastes a day.",
  "features":{"steps":3,"crews":3,"agents":3,"tasks":3,"tools":2,"parallel":2,"models":1},
  "controls":{"structured":"KeyPoints are extracted once and validated.","deterministic":"Code counts the tweets, their characters and the words of the post.","grounding":"Every writer receives only the extracted key points."},
  "why":"Three Crews write in parallel and an and_ step waits for all of them: complexity 6. Precision is 6: the key points are validated, the writers receive only those points, and code checks tweet and post length. No agent checks that the posts stay faithful, so the project sits just above the line.",
  "steps":{"3":"PackState holds the article, the key points and the three outputs.","4":"@start() load_article reads a web address with ScrapeWebsiteTool, or uses the pasted text.",
           "5":"extract_points is a direct LLM call. Three listeners on it start three Crews side by side, and and_ waits for all three. publish_pack then saves three files.","6":"kickoff(inputs={article}).","7":"The LinkedIn post, the six tweets, the newsletter and code checks of their length."}},
 {"n":9,"title":"Lead Qualification Flow","kind":"Flow","tags":["Flow + Crews","Router"],
  "problem":"Sales teams waste time on cold leads while hot leads wait days for a reply.",
  "features":{"steps":4,"routers":1,"crews":2,"agents":2,"tasks":2,"tools":2,"models":2},
  "controls":{"structured":"CompanyProfile and LeadScore are validated.","deterministic":"An unreadable website is rated cold by code, without asking the model.","grounding":"ScrapeWebsiteTool reads the company's own site.","routing":"The router returns only hot, warm or cold."},
  "why":"An enrichment Crew reads the website, a router rates the lead, and only a hot lead receives an outreach Crew: complexity 5. When the website cannot be read, code rates the lead cold without asking the model, so there is no guess without evidence. Precision is 7.",
  "steps":{"3":"LeadState holds the company, its profile, the rating and the action.","4":"@start() capture_lead.",
           "5":"enrich_crew reads the site. The router rates the lead hot, warm or cold. Hot leads get an outreach Crew, and or_ runs log_lead, which adds a row to leads.csv.","6":"kickoff(inputs={name, website, offer}).","7":"The rating, points, reasons, action and, for hot leads, a drafted email."}},
 {"n":10,"title":"Approval Report Flow","kind":"Flow","tags":["Flow + Crews","Human feedback"],
  "problem":"Auto-generated reports are risky to publish unchecked, but reviewing every word by hand is slow.",
  "features":{"steps":2,"routers":1,"crews":3,"agents":5,"tasks":5,"tools":1,"human":1,"loop":1,"models":1},
  "controls":{"structured":"ResearchPack validates each claim together with its source.","deterministic":"Code decides publish or revise from the person's decision.","grounding":"The Research Crew searches the web and keeps the source of every claim.","verification":"A Fact Checker deletes any claim that has no source.","routing":"The router returns only approved or revise.","human":"A person approves the draft before the final text is set."},
  "why":"The only project that uses every safeguard, including a person's approval before anything is published: precision 10. It also costs the most complexity, 9: five agents, three Crews, a router, a human gate and a revision loop. This region is used because a published report is public, so the person has the last word. A score of 10 is the design maximum and not a measure of accuracy.",
  "steps":{"3":"ApprovalReportState holds the topic, findings, draft, decision and status.","4":"@start() collect_topic.",
           "5":"@router check_decision first checks whether a decision has arrived. New reports go through a Research Crew and a Writing Crew, then review_draft, which carries the @human_feedback decorator, pauses for the person. approved runs publish_report, which saves final.md. revise runs the Revision Crew and returns to review_draft. The pause is the gap between two calls.","6":"kickoff runs once per call: first without a decision, then with it.","7":"status awaiting_review with a draft, or published with the final text."}},
]

def clamp(value,low,high):
  return max(low,min(high,value))

def score_project(p):
  c_points=sum(p["features"].get(key,0)*weight for key,_,weight,_ in WEIGHTS)
  p_points=sum(points for key,_,points,_ in SAFEGUARDS if key in p["controls"])
  c_score=clamp(int(c_points/4+0.5),1,10)           # complexity points divided by 4, rounded, kept between 1 and 10
  p_score=clamp(int(p_points/10+0.5),1,10)          # precision points divided by 10, rounded, kept between 1 and 10
  return c_points,p_points,c_score,p_score

def split_source():
  #---reads this file so that the website shows the real code, cut into the seven steps
  try:
    lines=open(__file__,encoding="utf-8").read().splitlines()
  except Exception:
    return {},[]
  shared=[]
  begin=next((i for i,l in enumerate(lines) if l.startswith("#---Step 1:")),None)
  end=next((i for i,l in enumerate(lines) if l.startswith("#---end of steps 1 and 2")),None)
  if begin is not None and end is not None:
    shared=lines[begin:end]
  banners=[i for i,l in enumerate(lines) if re.match(r"^# (\d+\. |END OF THE TEN)",l)]
  blocks={}
  for k in range(len(banners)-1):
    block=lines[banners[k]+2:banners[k+1]-1]
    while block and not block[-1].strip():
      block.pop()
    blocks[k+1]=block
  return blocks,shared

def code_for(n,blocks,shared):
  code=shared+[""]+blocks.get(n,[])
  marks=[(i,int(m.group(1)),m.group(2).strip()) for i,l in enumerate(code) for m in [re.match(r"^\s*#---Step (\d): ([^(]+)",l)] if m]
  steps=[]
  for k,(i,number,name) in enumerate(marks):
    stop=marks[k+1][0] if k+1<len(marks) else len(code)
    for j in range(i+1,stop):
      if code[j].startswith("@app.") :
        stop=j
        break
    while stop>i+1 and not code[stop-1].strip():
      stop-=1
    steps.append({"n":number,"name":name,"start":i,"end":stop-1})
  return code,steps

SAMPLE_VALUES={
 1:[{"label":"AI in farming","values":{"topic":SAMPLES["topic"]}},{"label":"Home battery storage","values":{"topic":"Home battery storage"}}],
 2:[{"label":"Programming languages","values":{"competitors":"\n".join(SAMPLES["competitors"]),"your_product":SAMPLES["your_product"]}}],
 3:[{"label":"Product meeting","values":{"transcript":SAMPLES["transcript"]}}],
 4:[{"label":"Four weeks of sales","values":{"csv_text":SAMPLES["csv"]}}],
 5:[{"label":t["label"],"values":{"ticket_id":t["ticket_id"],"customer_name":t["customer_name"],"issue_description":t["issue_description"]}} for t in SAMPLES["tickets"]],
 6:[{"label":"Data analyst role","values":{"job_description":SAMPLES["job"],"resumes":"\n=====\n".join(r["name"]+"\n"+r["text"] for r in SAMPLES["resumes"])}}],
 7:[{"label":i["label"],"values":{"title":i["title"],"body":i["body"],"docs_url":i.get("docs_url","")}} for i in SAMPLES["issues"]],
 8:[{"label":"Team decisions","values":{"article":SAMPLES["article"],"url":""}}],
 9:[{"label":l["label"],"values":{"name":l["name"],"website":l["website"],"offer":l["offer"]}} for l in SAMPLES["leads"]],
 10:[{"label":"Remote work","values":SAMPLES["report"]}],
}
FIELDS={
 1:[{"name":"topic","label":"Topic"}],
 2:[{"name":"competitors","label":"Competitor websites","kind":"area","rows":3,"as":"urls","hint":"One address per line, at most three."},{"name":"your_product","label":"Your product"}],
 3:[{"name":"transcript","label":"Meeting transcript","kind":"area","rows":9}],
 4:[{"name":"csv_text","label":"Sales data (CSV)","kind":"area","rows":9,"hint":"Columns: date, region, product, units, unit_price."}],
 5:[{"name":"ticket_id","label":"Ticket id"},{"name":"customer_name","label":"Customer name"},{"name":"issue_description","label":"Issue description","kind":"area","rows":4}],
 6:[{"name":"job_description","label":"Job description","kind":"area","rows":4},{"name":"resumes","label":"Resumes","kind":"area","rows":9,"as":"resumes","hint":"First line is the name. Separate resumes with a line of =====."}],
 7:[{"name":"title","label":"Issue title"},{"name":"body","label":"Issue text","kind":"area","rows":4},{"name":"docs_url","label":"Documentation page (optional)","optional":True,"hint":"Used only to answer questions."}],
 8:[{"name":"url","label":"Article web address (optional)","optional":True,"hint":"If you fill this in, the page is read with ScrapeWebsiteTool and the text below is ignored."},{"name":"article","label":"Article text","kind":"area","rows":9}],
 9:[{"name":"name","label":"Company"},{"name":"website","label":"Website"},{"name":"offer","label":"What you sell"}],
 10:[{"name":"topic","label":"Report topic"},{"name":"audience","label":"Audience"}],
}

PLAN={
 1:["Researcher gathers sources","Writer drafts the post","Editor tightens the draft"],
 2:["Manager reads the brief","Site Scraper, Pricing Analyst and News Analyst are assigned work","Strategist writes the comparison"],
 3:["Summariser, Action Finder and Decision Logger read the transcript together","Formatter joins their outputs"],
 4:["pandas tool reads the CSV","Data Analyst selects the findings","Insight Writer explains them","Reviewer checks the numbers"],
 5:["Ticket is received","Model categorises it","Router chooses billing, account access or other","Reply is written"],
 6:["Resume is loaded","Model scores it","Python threshold routes it","Summary Crew or rejection draft"],
 7:["Issue is read","Model classifies it","Router picks a specialist Crew","Bug, Feature or Answer Crew responds"],
 8:["Article is loaded","Key points are extracted","LinkedIn, Twitter and Newsletter Crews run in parallel","and_ waits for all three"],
 9:["Lead is captured","Researcher enriches the lead","Router scores hot, warm or cold","Outreach, nurture or archive"],
 10:["Topic is collected","Router checks for a decision","Research Crew: Researcher, Analyst, Fact Checker","Writer produces a draft","You approve or request changes"],
}

def site_data():
  blocks,shared=split_source()
  out=[]
  for p in PROJECTS:
    n=p["n"]
    c_points,p_points,c_score,p_score=score_project(p)
    code,steps=code_for(n,blocks,shared)
    crew=p["kind"]=="Crew"
    names=STEPS_CREW if crew else STEPS_FLOW
    notes=[{"n":i+1,"name":names[i][0],"note":names[i][1],"here":p["steps"].get(str(i+1),"")} for i in range(7)]
    out.append({"n":n,"title":p["title"],"kind":p["kind"],"tags":p["tags"],"problem":p["problem"],"why":p["why"],
                "cPoints":c_points,"pPoints":p_points,"c":c_score,"p":p_score,"quadrant":QUADRANTS[(c_score>=5,p_score>=5)],
                "complexity":[{"label":label,"count":p["features"][key],"weight":weight,"why":why} for key,label,weight,why in WEIGHTS if p["features"].get(key)],
                "precision":[{"label":label,"points":points,"why":why,"present":key in p["controls"],"evidence":p["controls"].get(key,"")} for key,label,points,why in SAFEGUARDS],
                "steps":notes,"code":code,"ranges":steps,"fields":FIELDS[n],"samples":SAMPLE_VALUES[n],"reviewing":n==10,"plan":PLAN[n]})
  return {"projects":out}

_page={}
def build_page():
  if "html" not in _page:
    d=site_data()
    data=json.dumps(d,ensure_ascii=False).replace("</","<\\/")
    page=SITE_HTML.replace("/*RAIL*/",render_rail(d)).replace("/*HERO*/",render_hero(d)).replace("/*SECTIONS*/",render_sections(d)).replace("/*FOOT*/",render_foot(d))
    page=page.replace("/*CSS*/",SITE_CSS).replace("/*DATA*/",data).replace("/*JS*/",SITE_JS)
    _page["html"]=page
  return _page["html"]

@app.get("/",response_class=HTMLResponse,include_in_schema=False)
def website():
  return build_page()


LOGO_SVG=r'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" width="240" height="240" role="img" aria-label="CREWAI mark: ten nodes in a ring around one square"><title>CREWAI</title><path d="M191.01 161.00 A82 82 0 1 1 191.01 79.00" fill="none" stroke="#101010" stroke-width="2.20" stroke-linecap="round"/><line x1="191.01" y1="79.00" x2="134.00" y2="120.00" stroke="#101010" stroke-width="2.20" stroke-linecap="round"/><circle cx="191.01" cy="161.00" r="7.00" fill="#101010"/><circle cx="156.80" cy="193.28" r="7.62" fill="#101010"/><circle cx="110.48" cy="201.45" r="8.24" fill="#101010"/><circle cx="67.29" cy="182.82" r="8.86" fill="#101010"/><circle cx="41.44" cy="143.52" r="9.48" fill="#101010"/><circle cx="41.44" cy="96.48" r="10.10" fill="#101010"/><circle cx="67.29" cy="57.18" r="10.72" fill="#101010"/><circle cx="110.48" cy="38.55" r="11.34" fill="#101010"/><circle cx="156.80" cy="46.72" r="11.96" fill="#101010"/><circle cx="191.01" cy="79.00" r="12.58" fill="#FF4B1F"/><rect x="103.00" y="103.00" width="34.00" height="34.00" rx="5.00" fill="#101010"/></svg>'''

def esc(text):
  return html.escape(str(text),quote=True)

QUADRANT_ORDER=["Simple Crews","Complex Crews","Flows with direct LLM calls","Flows orchestrating Crews"]
QUADRANT_RULE={"Simple Crews":("low","low"),"Complex Crews":("high","low"),"Flows with direct LLM calls":("low","high"),"Flows orchestrating Crews":("high","high")}
ARROW_SVG='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>'

def render_rail(d):
  links="".join(f'<a href="#p{p["n"]}" data-n="{p["n"]}" style="--q:var(--q{QUADRANT_ORDER.index(p["quadrant"])})" aria-label="Project {p["n"]}: {esc(p["title"])}"><span>{p["n"]:02d}</span><span class="tip">{esc(p["title"])}</span></a>' for p in d["projects"])
  return f'<nav class="rail" aria-label="Projects"><a class="brand" href="#top" aria-label="CREWAI, top of page"><span class="lg">{LOGO_SVG}</span>CREWAI</a>{links}</nav>'

def ticker(projects):
  one="".join(f'<a href="#p{p["n"]}" tabindex="-1"><i style="--q:var(--q{QUADRANT_ORDER.index(p["quadrant"])})"></i>/agent{p["n"]} · {esc(p["title"])}</a>' for p in projects)
  return f'<div class="ticker" aria-hidden="true"><div>{one}{one}</div></div>'

def render_hero(d):
  projects=d["projects"]
  crews=sum(1 for p in projects if p["kind"]=="Crew")
  legend=""
  for qi_,name in enumerate(QUADRANT_ORDER):
    members=" · ".join(f'{p["n"]:02d}' for p in projects if p["quadrant"]==name) or "none"
    legend+=f'<button type="button" data-q="{qi_}" style="--q:var(--q{qi_})"><i></i><div><b>{esc(name)}</b><span>{members}</span></div></button>'
  return (f'<header class="hero"><div class="page"><div class="hl"><span class="lg big rise" aria-hidden="true">{LOGO_SVG}</span><span class="kicker rise">CrewAI · FastAPI · Gemini</span>'
          f'<h1 class="rise" style="--i:1">Ten agents,<br>one <span class="u">server</span>.</h1>'
          f'<p class="lede rise" style="--i:2">Each project is scored on complexity and precision, explained in the seven steps of the notes with its real code, and can be run on this page. Every agent is also a route: /agent1 to /agent10.</p>'
          f'<div class="stats rise" style="--i:3"><span><b>10</b> routes</span><span><b>{crews}</b> Crews</span><span><b>{10-crews}</b> Flows</span><span><b>7</b> steps each</span></div>'
          f'<div class="cta rise" style="--i:4"><a class="btn-run" href="#p1">Start with project 1 {ARROW_SVG}</a><a class="btn-line" href="/docs">Open the API docs</a><a class="lnk" href="/agent1">GET /agent1</a></div></div>'
          f'<div class="hr rise" style="--i:3"><div class="gcard"><div class="gfx" id="master"></div><div class="tipbox" id="tip"></div></div><div class="legend">{legend}</div></div></div></header>'
          +ticker(projects))

def ledger_complexity(p):
  total=max(p["cPoints"],1)
  rows="".join(f'<div class="row" style="--k:{k};--w:{round(100*r["count"]*r["weight"]/total)}%"><b>{esc(r["label"])}</b><em>{r["count"]} × {r["weight"]} = {r["count"]*r["weight"]}</em><small>{esc(r["why"])}</small><div class="meter"><i></i></div></div>' for k,r in enumerate(p["complexity"]))
  raw=p["cPoints"]/4
  return (f'<div class="led"><h4>Complexity: how the points were given</h4>{rows}'
          f'<div class="conv"><span>{p["cPoints"]} points</span><i>÷ 4</i><span>= {raw:.2f}</span><i>→</i><span>score {p["c"]}</span><small>rounded and kept between 1 and 10</small></div></div>')

def ledger_precision(p):
  rows=""
  for k,r in enumerate(p["precision"]):
    if r["present"]:
      rows+=f'<div class="row" style="--k:{k};--w:{round(100*r["points"]/20)}%"><b>{esc(r["label"])}</b><em>+{r["points"]}</em><small class="ev">{esc(r["evidence"])}</small><div class="meter"><i></i></div></div>'
    else:
      rows+=f'<div class="row off" style="--k:{k};--w:0%"><b>{esc(r["label"])}</b><em>0</em><small>Not used in this project. {esc(r["why"])}</small></div>'
  raw=p["pPoints"]/10
  return (f'<div class="led"><h4>Precision: how the points were given</h4>{rows}'
          f'<div class="conv"><span>{p["pPoints"]} points</span><i>÷ 10</i><span>= {raw:.2f}</span><i>→</i><span>score {p["p"]}</span><small>rounded and kept between 1 and 10</small></div></div>')

def render_sections(d):
  out=[]
  for p in d["projects"]:
    n=p["n"]
    qi_=QUADRANT_ORDER.index(p["quadrant"])
    cl,pl=QUADRANT_RULE[p["quadrant"]]
    steps="".join(f'<li><b>Step {s["n"]} · {esc(s["name"])}</b> {esc(s["note"])}<br>{esc(s["here"])}</li>' for s in p["steps"])
    out.append(
      f'<section class="proj" id="p{n}" data-n="{n}" style="--q:var(--q{qi_})"><div class="page">'
      f'<header class="ph rv"><div class="pnum" aria-hidden="true">{n:02d}</div><div><span class="pkick">{esc(" · ".join(p["tags"]))}</span><h2>{esc(p["title"])}</h2><p class="problem">{esc(p["problem"])}</p></div></header>'
      f'<div class="blk"><h3 class="bh rv"><span>A</span>Where it sits on the graph</h3>'
      f'<div class="score-grid"><div class="gcard rv"><div class="gfx" data-n="{n}"></div></div>'
      f'<div class="sread rv"><div class="nums"><div class="num c"><span>Complexity</span><b data-count="{p["c"]}">{p["c"]}</b><small>out of 10</small></div><div class="num"><span>Precision</span><b data-count="{p["p"]}">{p["p"]}</b><small>out of 10</small></div></div>'
      f'<p class="region"><b>{esc(p["quadrant"])}</b><span>Complexity {p["c"]} is {cl} and precision {p["p"]} is {pl}. On the notes\' graph, 1 to 4 is low and 5 to 10 is high.</span></p>'
      f'<p class="why">{esc(p["why"])}</p></div></div>'
      f'<div class="ledgers rv">{ledger_complexity(p)}{ledger_precision(p)}</div></div>'
      f'<div class="blk"><h3 class="bh rv"><span>B</span>The seven steps, in the code</h3><div class="stepper rv" data-n="{n}"></div><ol class="stepslist">{steps}</ol></div>'
      f'<div class="blk"><h3 class="bh rv"><span>C</span>Run it</h3><div class="try rv" data-n="{n}"></div>'
      f'<p class="ends"><span>Open in a browser:</span><a class="lnk" href="/agent{n}">GET /agent{n}</a><a class="lnk" href="/agent{n}?run=true">GET /agent{n}?run=true</a></p></div>'
      f'</div></section>')
  return "".join(out)

def render_foot(d):
  links=" ".join(f'<a class="lnk" href="/agent{i}">/agent{i}</a>' for i in range(1,11))
  return (f'<footer class="foot"><div class="page"><p>The code panels read this server\'s own app.py, and the scores are calculated from the counts in the ledgers, so the page and the code cannot disagree. '
          f'The scores describe how each project is designed. They are not measurements of accuracy.</p><p class="ends">{links} <a class="lnk" href="/docs">/docs</a> <a class="lnk" href="/health">/health</a></p></div></footer>')

SITE_CSS=r""":root{color-scheme:light;--ink:#14141A;--ink2:#3A3A44;--muted:#666672;--coral:#FF5A50;--coral-d:#C93A31;--coral-l:#FFE3E0;--tint:#F6F6F7;--line:#E4E4E7;--ok:#1F7A4D;--ok-l:#DDF1E6;--warn:#8A5A00;--warn-l:#FFF1D6;
--q0:#5B6270;--q1:#B45309;--q2:#0E7C86;--q3:#C93A31;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Courier New",monospace;--ease:cubic-bezier(.2,.8,.2,1)}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scroll-padding-top:84px;background:#fff}
body{font-family:Calibri,"Segoe UI",system-ui,-apple-system,sans-serif;color:var(--ink);background:#fff;line-height:1.5;-webkit-font-smoothing:antialiased;overflow-x:hidden}
.mono{font-family:var(--mono)}
:focus-visible{outline:3px solid var(--ink);outline-offset:3px;border-radius:6px}
a{color:inherit}
.bar{position:fixed;left:0;top:0;height:3px;width:100%;background:var(--coral);transform:scaleX(0);transform-origin:0 50%;z-index:60}
.page{max-width:1120px;margin:0 auto;padding:0 clamp(20px,5vw,56px)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

/* rail */
.rail{position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:6px;padding:10px clamp(12px,3vw,28px);background:rgba(255,255,255,.92);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);overflow-x:auto;scrollbar-width:none}
.rail::-webkit-scrollbar{display:none}
.rail .brand{font-weight:700;letter-spacing:-.01em;margin-right:8px;white-space:nowrap}
.rail .brand b{color:var(--coral)}
.rail a{position:relative;flex:0 0 auto;display:grid;place-items:center;min-width:44px;height:44px;border-radius:12px;font:700 .9rem var(--mono);text-decoration:none;color:var(--ink2);border:1.5px solid transparent;transition:background .25s,color .25s,border-color .25s,transform .3s var(--ease)}
.rail a::after{content:"";position:absolute;left:50%;bottom:5px;width:5px;height:5px;margin-left:-2.5px;border-radius:50%;background:var(--q,#ccc);opacity:.9}
.rail a:hover{background:var(--tint);transform:translateY(-2px)}
.rail a.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.rail a .tip{display:none}
@media (min-width:1240px){
  .rail{position:fixed;left:14px;top:50%;transform:translateY(-50%);flex-direction:column;gap:4px;padding:10px 8px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.9);overflow:visible;box-shadow:0 10px 30px rgba(20,20,26,.06)}
  .rail .brand{display:none}
  .rail a{min-width:44px}
  .rail a .tip{display:block;position:absolute;left:calc(100% + 12px);top:50%;transform:translate(-6px,-50%);background:var(--ink);color:#fff;font:700 .8rem Calibri,"Segoe UI",system-ui,sans-serif;padding:6px 10px;border-radius:8px;white-space:nowrap;opacity:0;pointer-events:none;transition:opacity .2s,transform .3s var(--ease)}
  .rail a:hover .tip,.rail a:focus-visible .tip{opacity:1;transform:translate(0,-50%)}
}
.pill{position:fixed;right:14px;bottom:14px;z-index:55;display:inline-flex;align-items:center;gap:8px;background:#fff;border:1.5px solid var(--line);border-radius:999px;padding:9px 16px;min-height:44px;font:700 .88rem Calibri,"Segoe UI",system-ui,sans-serif;color:var(--ink);box-shadow:0 8px 24px rgba(20,20,26,.1)}
.pill i{width:10px;height:10px;border-radius:50%;background:#b8b8bf;display:inline-block}
.pill.ok i{background:var(--ok);box-shadow:0 0 0 4px var(--ok-l)}.pill.bad i{background:var(--coral)}.pill.wait i{background:var(--warn);animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.25}}

/* hero */
.hero{position:relative;padding:clamp(48px,9vh,96px) 0 clamp(40px,7vh,72px);background-image:radial-gradient(var(--line) 1.2px,transparent 1.2px);background-size:26px 26px}
.hero::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,#fff 22%,rgba(255,255,255,.55) 60%,rgba(255,255,255,0) 100%);pointer-events:none}
.hero .page{position:relative;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);gap:clamp(24px,5vw,64px);align-items:center}
.kicker{display:inline-flex;align-items:center;gap:10px;font-size:.8rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--coral-d)}
.kicker::before{content:"";width:34px;height:2px;background:var(--coral);border-radius:2px;transform-origin:0 50%;animation:grow .8s .2s var(--ease) both}
@keyframes grow{from{transform:scaleX(0)}to{transform:none}}
h1{font-size:clamp(2.6rem,6.4vw,5rem);line-height:1;letter-spacing:-.035em;margin:16px 0 18px;font-weight:700}
h1 .u{position:relative;display:inline-block;white-space:nowrap}
h1 .u::after{content:"";position:absolute;left:0;right:0;bottom:.04em;height:.14em;background:var(--coral);border-radius:3px;transform-origin:0 50%;animation:grow .9s .7s var(--ease) both}
.lede{font-size:clamp(1.05rem,1.8vw,1.3rem);color:var(--ink2);max-width:520px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}
.stats span{font-size:.92rem;font-weight:700;border:1.5px solid var(--line);border-radius:999px;padding:6px 14px;background:#fff}
.stats b{color:var(--coral-d);font-family:var(--mono)}
.cta{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:center}
.rise{animation:rise .8s var(--ease) both;animation-delay:calc(var(--i,0)*90ms)}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}

/* buttons: every kind has its own look */
.btn-run{position:relative;overflow:hidden;display:inline-flex;align-items:center;justify-content:center;gap:12px;min-height:50px;padding:0 26px;border:0;border-radius:14px;background:var(--coral);color:#fff;font:700 1rem Calibri,"Segoe UI",system-ui,sans-serif;cursor:pointer;text-decoration:none;transition:background .25s,transform .25s var(--ease),box-shadow .3s}
.btn-run::after{content:"";position:absolute;top:0;bottom:0;left:-60%;width:40%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.4),transparent);transform:skewX(-18deg);transition:left .7s var(--ease)}
.btn-run:hover{background:var(--coral-d);box-shadow:0 12px 26px rgba(201,58,49,.3);transform:translateY(-2px)}
.btn-run:hover::after{left:130%}
.btn-run:active{transform:scale(.97)}
.btn-run svg{width:18px;height:18px;stroke:#fff;fill:none;stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round;transition:transform .3s var(--ease)}
.btn-run:hover svg{transform:translateX(5px)}
.btn-run[disabled]{cursor:progress;background:var(--ink);box-shadow:none;transform:none}
.btn-run[disabled] svg{display:none}
.btn-run .sp{display:none;width:18px;height:18px;border-radius:50%;border:2.5px solid rgba(255,255,255,.3);border-top-color:#fff;animation:spin .8s linear infinite}
.btn-run[disabled] .sp{display:block}
@keyframes spin{to{transform:rotate(360deg)}}
.btn-line{position:relative;display:inline-flex;align-items:center;gap:10px;min-height:50px;padding:0 22px;border-radius:14px;border:1.5px solid var(--ink);background:#fff;color:var(--ink);font:700 1rem Calibri,"Segoe UI",system-ui,sans-serif;text-decoration:none;overflow:hidden;z-index:0;cursor:pointer;transition:color .3s,transform .25s var(--ease)}
.btn-line::before{content:"";position:absolute;inset:0;background:var(--ink);transform:translateY(101%);transition:transform .4s var(--ease);z-index:-1}
.btn-line:hover{color:#fff;transform:translateY(-2px)}.btn-line:hover::before{transform:none}
.lnk{font:700 .95rem var(--mono);text-decoration:none;color:var(--ink);background:linear-gradient(var(--coral),var(--coral)) 0 100%/0 2px no-repeat;padding-bottom:3px;transition:background-size .35s var(--ease),color .2s}
.lnk:hover{background-size:100% 2px;color:var(--coral-d)}
.chip{font:700 .88rem Calibri,"Segoe UI",system-ui,sans-serif;color:var(--ink2);background:#fff;border:1.5px solid var(--line);border-radius:999px;padding:7px 16px;min-height:40px;cursor:pointer;transition:border-color .2s,background .2s,color .2s,transform .25s var(--ease)}
.chip:hover{border-color:var(--coral);color:var(--coral-d);transform:translateY(-2px)}
.chip[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:#fff}
.copy{display:inline-flex;align-items:center;gap:8px;font:700 .8rem var(--mono);color:var(--ink);background:#fff;border:1.5px solid var(--ink);border-radius:8px;padding:7px 12px;min-height:38px;cursor:pointer;transition:background .2s,color .2s,transform .2s var(--ease)}
.copy svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.copy .ck{display:none}.copy:hover{background:var(--ink);color:#fff;transform:translateY(-1px)}.copy:active{transform:scale(.96)}
.copy.ok{background:var(--ok);border-color:var(--ok);color:#fff}.copy.ok .cp{display:none}.copy.ok .ck{display:block}
.yes{position:relative;min-height:46px;padding:0 22px;border-radius:12px;border:1.5px solid var(--ink);background:var(--ink);color:#fff;font:700 1rem Calibri,"Segoe UI",system-ui,sans-serif;cursor:pointer;overflow:hidden;transition:transform .25s var(--ease),box-shadow .3s}
.yes::after{content:"";position:absolute;left:18px;right:18px;bottom:8px;height:2px;background:var(--coral);transform:scaleX(0);transform-origin:0 50%;transition:transform .4s var(--ease)}
.yes:hover{transform:translateY(-2px);box-shadow:0 10px 22px rgba(20,20,26,.2)}.yes:hover::after{transform:none}
.no{position:relative;min-height:46px;padding:0 22px;border-radius:12px;border:1.5px solid var(--coral);background:#fff;color:var(--coral-d);font:700 1rem Calibri,"Segoe UI",system-ui,sans-serif;cursor:pointer;overflow:hidden;z-index:0;transition:color .3s,transform .25s var(--ease)}
.no::before{content:"";position:absolute;inset:0;background:var(--coral);transform:translateX(-101%);transition:transform .45s var(--ease);z-index:-1}
.no:hover{color:#fff;transform:translateY(-2px)}.no:hover::before{transform:none}
.yes[disabled],.no[disabled]{opacity:.5;cursor:progress;transform:none}
.ctl{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:42px;min-width:42px;padding:0 14px;border-radius:999px;border:1.5px solid var(--line);background:#fff;font:700 .9rem Calibri,"Segoe UI",system-ui,sans-serif;color:var(--ink);cursor:pointer;transition:border-color .2s,background .2s,transform .25s var(--ease)}
.ctl:hover:not([disabled]){border-color:var(--ink);transform:translateY(-1px)}
.ctl[disabled]{opacity:.4;cursor:default}
.ctl.play{background:var(--tint)}.ctl.play[aria-pressed=true]{background:var(--coral);border-color:var(--coral);color:#fff}
.ctl svg{width:14px;height:14px;fill:currentColor}

/* hero graph */
.gcard{position:relative;background:#fff;border:1px solid var(--line);border-radius:20px;padding:clamp(12px,2vw,20px);box-shadow:0 20px 50px rgba(20,20,26,.06)}
.gfx svg{display:block;width:100%;height:auto;overflow:visible}
.gfx text{font-family:Calibri,"Segoe UI",system-ui,sans-serif}
.gfx .gl{stroke:var(--line);stroke-width:1}
.gfx .dv{stroke:var(--muted);stroke-width:1.25;stroke-dasharray:5 5}
.gfx .tk{font-size:11.5px;fill:#8a8a94}
.gfx .ax{font-size:12.5px;font-weight:700;fill:var(--muted);letter-spacing:.04em}
.gfx .ql{font-size:11.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;fill:#9a9aa4;transition:fill .3s}
.gfx .ql.hot{fill:var(--q)}
.gfx .qa{opacity:0;transition:opacity .8s ease .2s}
.gfx.in .qa{opacity:.12}
.gfx .pt{cursor:pointer;transition:transform .3s var(--ease)}
.gfx .pt circle{transition:r .3s var(--ease),fill .2s}
.gfx .pt:hover circle,.gfx .pt:focus circle{r:15}
.gfx .pt text{font-size:12px;font-weight:700;fill:#fff;pointer-events:none}
.gfx.faint .pt:not(.me) circle{fill:#d5d5db!important}
.gfx.faint .pt:not(.me) text{fill:var(--ink2)}
.gfx .pt{opacity:0;transform:scale(.3);transform-box:fill-box;transform-origin:center}
.gfx.in .pt{opacity:1;transform:none;transition:opacity .5s ease calc(var(--i)*70ms+.3s),transform .6s var(--ease) calc(var(--i)*70ms+.3s)}
.gfx .ring{fill:none;stroke:var(--q);stroke-width:2;transform-box:fill-box;transform-origin:center;opacity:0}
.gfx.in .ring{animation:ring 2.2s ease-out 1.5s infinite}
@keyframes ring{0%{transform:scale(.8);opacity:.7}100%{transform:scale(2.6);opacity:0}}
.gfx .cross{stroke:var(--q);stroke-width:1.6;stroke-dasharray:200;stroke-dashoffset:200;transition:stroke-dashoffset 1s var(--ease) .9s}
.gfx.in .cross{stroke-dashoffset:0}
.gfx .lab{font-size:13.5px;font-weight:700;fill:var(--q);opacity:0;transition:opacity .5s ease 1.6s}
.gfx.in .lab{opacity:1}
.tipbox{position:absolute;z-index:5;pointer-events:none;background:var(--ink);color:#fff;border-radius:10px;padding:8px 12px;font-size:.85rem;line-height:1.3;opacity:0;transform:translateY(4px);transition:opacity .2s,transform .25s var(--ease);max-width:220px}
.tipbox.on{opacity:1;transform:none}.tipbox b{display:block}.tipbox span{color:#d8d8de}
.legend{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.legend button{display:flex;align-items:flex-start;gap:10px;text-align:left;border:1.5px solid var(--line);background:#fff;border-radius:12px;padding:9px 12px;font:inherit;cursor:pointer;transition:border-color .2s,transform .25s var(--ease),background .2s}
.legend button:hover,.legend button.on{border-color:var(--q);background:var(--tint);transform:translateY(-2px)}
.legend i{flex:0 0 12px;height:12px;margin-top:5px;border-radius:4px;background:var(--q)}
.legend b{display:block;font-size:.9rem;line-height:1.2}.legend span{font:.8rem var(--mono);color:var(--muted)}

/* project */
.proj{padding:clamp(56px,9vh,104px) 0;border-top:1px solid var(--line)}
.ph{display:grid;grid-template-columns:auto 1fr;gap:clamp(14px,3vw,34px);align-items:start}
.pnum{font:700 clamp(3.4rem,9vw,7rem)/.85 var(--mono);letter-spacing:-.06em;color:transparent;-webkit-text-stroke:2px var(--q);transition:color .8s ease .2s}
.rv.in .pnum{color:var(--q)}
.pkick{font-size:.8rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--q)}
h2{font-size:clamp(1.9rem,4.4vw,3.1rem);line-height:1.08;letter-spacing:-.025em;margin:6px 0 10px}
.problem{font-size:clamp(1.05rem,2vw,1.3rem);color:var(--ink2);max-width:760px}
.blk{margin-top:clamp(30px,5vh,52px)}
.bh{display:flex;align-items:center;gap:12px;font-size:1.1rem;font-weight:700;margin-bottom:16px}
.bh span{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;background:var(--q);color:#fff;font:700 .85rem var(--mono)}
.bh::after{content:"";flex:1;height:1px;background:var(--line)}
.score-grid{display:grid;grid-template-columns:minmax(0,.92fr) minmax(0,1.08fr);gap:clamp(18px,3vw,34px);align-items:start}
.nums{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.num{background:var(--tint);border:1px solid var(--line);border-radius:16px;padding:12px 18px}
.num span{display:block;font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.num b{display:block;font:700 clamp(2.6rem,5vw,3.6rem)/1.05 var(--mono);letter-spacing:-.05em}
.num.c b{color:var(--q)}
.num small{color:var(--muted);font-size:.85rem}
.region{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;margin-top:14px}
.region b{background:var(--q);color:#fff;border-radius:8px;padding:4px 12px;font-size:.95rem}
.region span{color:var(--ink2);font-size:.95rem}
.why{margin-top:14px;font-size:1.03rem;line-height:1.55;color:var(--ink2);border-left:4px solid var(--q);padding-left:16px}
.ledgers{display:grid;grid-template-columns:1fr 1fr;gap:clamp(14px,2vw,24px);margin-top:18px}
.led{background:var(--tint);border:1px solid var(--line);border-radius:16px;padding:clamp(14px,2vw,20px)}
.led h4{font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
.row{display:grid;grid-template-columns:1fr auto;gap:2px 12px;padding:9px 0;border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.row b{font-size:.96rem}.row em{font:700 .88rem var(--mono);font-style:normal;text-align:right;white-space:nowrap}
.row small{grid-column:1/-1;color:var(--muted);font-size:.84rem;line-height:1.35}
.row .meter{grid-column:1/-1;height:6px;border-radius:6px;background:#e9e9ed;overflow:hidden;margin-top:4px}
.row .meter i{display:block;height:100%;width:var(--w);background:var(--q);border-radius:6px;transform:scaleX(0);transform-origin:0 50%;transition:transform 1s var(--ease) calc(var(--k)*90ms+.3s)}
.in .row .meter i{transform:none}
.row.off{opacity:.55}.row.off em{font-weight:400}
.row .ev{grid-column:1/-1;font-size:.86rem;color:var(--ink2)}
.conv{margin-top:12px;padding:12px 14px;background:#fff;border:1.5px dashed var(--q);border-radius:12px;font:700 .9rem var(--mono);color:var(--ink);display:flex;flex-wrap:wrap;gap:6px 10px;align-items:baseline}
.conv i{font-style:normal;color:var(--q)}
.conv small{font:.8rem Calibri,"Segoe UI",system-ui,sans-serif;color:var(--muted);font-weight:400}

/* stepper */
.stp-rail{display:flex;gap:0;overflow-x:auto;padding:4px 2px 12px;scrollbar-width:thin}
.stp-rail button{position:relative;flex:1 0 96px;display:flex;flex-direction:column;align-items:center;gap:6px;padding:4px 6px;background:none;border:0;cursor:pointer;font:700 .86rem Calibri,"Segoe UI",system-ui,sans-serif;color:var(--muted);transition:color .25s}
.stp-rail button::before{content:"";position:absolute;top:21px;left:-50%;width:100%;height:3px;background:var(--line);z-index:0}
.stp-rail button:first-child::before{display:none}
.stp-rail button::after{content:"";position:absolute;top:21px;left:-50%;width:100%;height:3px;background:var(--q);transform:scaleX(0);transform-origin:0 50%;transition:transform .5s var(--ease);z-index:0}
.stp-rail button.done::after,.stp-rail button.on::after{transform:none}
.stp-rail button:first-child::after{display:none}
.stp-rail i{position:relative;z-index:1;display:grid;place-items:center;width:44px;height:44px;border-radius:50%;background:#fff;border:2px solid var(--line);font:700 1rem var(--mono);font-style:normal;color:var(--ink2);transition:background .3s,border-color .3s,color .3s,transform .4s var(--ease)}
.stp-rail button:hover i{transform:translateY(-2px);border-color:var(--q)}
.stp-rail button.done i{border-color:var(--q);color:var(--q)}
.stp-rail button.on i{background:var(--q);border-color:var(--q);color:#fff;transform:scale(1.1)}
.stp-rail button.on{color:var(--ink)}
.stp-body{display:grid;grid-template-columns:minmax(0,.8fr) minmax(0,1.2fr);gap:clamp(14px,2vw,22px);align-items:stretch}
.card{background:var(--tint);border:1px solid var(--line);border-radius:16px;padding:clamp(16px,2.2vw,24px);display:flex;flex-direction:column;gap:12px}
.card .from{font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.card h4{font-size:1.5rem;letter-spacing:-.02em;line-height:1.1}
.card h4 em{font-style:normal;color:var(--q)}
.card .note{font-size:1.02rem;font-weight:700}
.card .here{font-size:1rem;color:var(--ink2);border-left:4px solid var(--q);padding-left:14px}
.card .swap{display:flex;flex-direction:column;gap:12px}
.card .ctrls{display:flex;flex-wrap:wrap;gap:8px;margin-top:auto;padding-top:6px}
.swap{animation:swap .45s var(--ease) both}
@keyframes swap{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.code{position:relative;background:#16161A;border-radius:16px;overflow:hidden;min-width:0}
.code .copy{position:absolute;right:10px;top:10px;z-index:2;background:#16161A;color:#fff;border-color:#4a4a52}
.code .copy:hover{background:#fff;color:var(--ink);border-color:#fff}
.code .cap{display:flex;align-items:center;gap:8px;padding:12px 16px;border-bottom:1px solid #2c2c34;font:700 .8rem var(--mono);color:#a8a8b3}
.code .cap i{width:10px;height:10px;border-radius:50%;background:#FF5A50;display:inline-block}
.code .cap i+i{background:#F4C06A}.code .cap i+i+i{background:#5BC98B}
.code .cap span{margin-left:6px}
.lines{max-height:470px;overflow:auto;padding:12px 0 16px;font:.82rem/1.6 var(--mono);color:#E8E8EC;scroll-behavior:smooth}
.lines::-webkit-scrollbar{width:10px;height:10px}.lines::-webkit-scrollbar-thumb{background:#44444c;border-radius:6px}
.ln{display:grid;grid-template-columns:44px 1fr;white-space:pre;opacity:.34;border-left:3px solid transparent;transition:opacity .35s,background .35s,border-color .35s}
.ln s{text-decoration:none;text-align:right;padding-right:12px;color:#6c6c78;user-select:none}
.ln.on{opacity:1;background:rgba(255,90,80,.1);border-left-color:var(--coral)}
.ln.on.first{animation:flash .9s ease}
@keyframes flash{from{background:rgba(255,90,80,.35)}}
.tc{color:#7d8590;font-style:italic}.ts{color:#FFB4AE}.tk2{color:#FF8A82;font-weight:700}.td{color:#F4C06A}.tf{color:#8FD0FF}.tn{color:#B8E2A0}
.stepslist{display:none}
html:not(.js) .stepslist{display:grid;gap:10px;list-style:none}
html:not(.js) .stepslist li{background:var(--tint);border:1px solid var(--line);border-radius:12px;padding:12px 16px}

/* run it */
.try{border:1px solid var(--line);border-radius:20px;background:#fff;overflow:hidden}
.tbar{display:flex;align-items:center;flex-wrap:wrap;gap:10px 14px;padding:12px 18px;background:var(--tint);border-bottom:1px solid var(--line)}
.tbar .m{font:700 .8rem var(--mono);background:var(--ink);color:#fff;border-radius:6px;padding:3px 8px}
.tbar .u{font:.9rem var(--mono);color:var(--ink2);word-break:break-all;flex:1;min-width:0}
.tbar a{font-size:.88rem;font-weight:700;color:var(--coral-d);text-decoration:none;background:linear-gradient(var(--coral),var(--coral)) 0 100%/0 2px no-repeat;transition:background-size .35s var(--ease);padding-bottom:2px}
.tbar a:hover{background-size:100% 2px}
.prog{height:3px;background:var(--line);overflow:hidden;visibility:hidden}
.prog i{display:block;height:100%;width:35%;background:var(--coral);border-radius:3px;animation:slide 1.3s ease-in-out infinite}
@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
.try.busy .prog{visibility:visible}
.tbody{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr)}
.form{padding:clamp(16px,2.4vw,24px);display:flex;flex-direction:column;gap:14px;border-right:1px solid var(--line)}
.fld{display:flex;flex-direction:column;gap:6px;min-width:0}
.fld>label,.flab{font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--coral-d)}
.inp{width:100%;font:1rem Calibri,"Segoe UI",system-ui,sans-serif;color:var(--ink);background:#fff;border:1.5px solid var(--line);border-radius:10px;padding:11px 14px;min-height:46px;transition:border-color .2s,box-shadow .2s}
textarea.inp{resize:vertical;line-height:1.4;min-height:110px;font-family:var(--mono);font-size:.86rem}
.inp:hover{border-color:#c9c9cf}.inp:focus{outline:none;border-color:var(--coral);box-shadow:0 0 0 4px rgba(255,90,80,.16)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.act{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px}
.hint{font-size:.85rem;color:var(--muted)}
.out{padding:clamp(16px,2.4vw,24px);min-height:260px;display:flex;flex-direction:column;gap:12px;min-width:0}
.empty{margin:auto;text-align:center;color:var(--muted);max-width:300px}
.empty svg{width:34px;height:34px;stroke:#cfcfd6;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;margin-bottom:8px}
.err{background:#FFF1F0;border:1px solid var(--coral-l);border-left:5px solid var(--coral);border-radius:12px;padding:12px 16px;color:#7A1E18;font-size:.95rem}
.err b{display:block;margin-bottom:2px}
.res{display:flex;flex-direction:column;gap:12px;animation:swap .5s var(--ease) both}
.took{font:700 .8rem var(--mono);color:var(--ok)}
.trail{display:flex;flex-wrap:wrap;align-items:center;gap:6px 4px}
.trail span{font:700 .78rem var(--mono);background:#fff;border:1.5px solid var(--q);color:var(--ink);border-radius:999px;padding:3px 10px;animation:swap .4s var(--ease) both;animation-delay:calc(var(--k)*110ms)}
.trail em{font-style:normal;color:var(--q);animation:swap .4s var(--ease) both;animation-delay:calc(var(--k)*110ms)}
.kv{display:flex;flex-direction:column;gap:3px}
.kv>.k{font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.kv>.v{font-size:.98rem}
.badge{display:inline-block;border-radius:999px;padding:3px 12px;font-size:.88rem;font-weight:700;background:var(--tint);border:1px solid var(--line)}
.badge.g{background:var(--ok-l);border-color:transparent;color:var(--ok)}.badge.r{background:#FFE3E0;border-color:transparent;color:var(--coral-d)}.badge.a{background:var(--warn-l);border-color:transparent;color:var(--warn)}
.text{white-space:pre-wrap;word-break:break-word;font-size:.95rem;line-height:1.55;background:var(--tint);border:1px solid var(--line);border-radius:12px;padding:12px 16px;max-height:420px;overflow:auto}
.text h5{font-size:1rem;margin:8px 0 2px}
.sub{border:1px solid var(--line);border-radius:12px;padding:12px 14px;display:flex;flex-direction:column;gap:10px}
.list{display:flex;flex-direction:column;gap:8px;list-style:none}
.list li{padding-left:16px;position:relative}.list li::before{content:"";position:absolute;left:0;top:.6em;width:7px;height:7px;border-radius:2px;background:var(--coral)}
.raw{font-size:.85rem;font-weight:700;color:var(--ink2);background:none;border:0;cursor:pointer;padding:4px 0;align-self:flex-start;text-decoration:underline;text-decoration-color:var(--coral);text-underline-offset:4px;text-decoration-thickness:2px}
.raw:hover{color:var(--coral-d)}
.rawpre{display:none;font:.8rem/1.5 var(--mono);background:#16161A;color:#E8E8EC;border-radius:12px;padding:12px 16px;overflow:auto;max-height:300px;white-space:pre-wrap;word-break:break-word}
.rawpre.open{display:block}
.review{display:flex;flex-direction:column;gap:10px;border-top:1px dashed var(--line);padding-top:12px}
.review .r2{display:flex;gap:10px;flex-wrap:wrap}
.ends{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin-top:14px;font-size:.92rem;color:var(--muted)}
.foot{padding:56px 0 110px;border-top:1px solid var(--line);color:var(--muted);font-size:.92rem}

/* reveal */
.js .rv{opacity:0;transform:translateY(24px);transition:opacity .7s ease,transform .8s var(--ease)}
.js .rv.in{opacity:1;transform:none}
@media (max-width:900px){
  .hero .page,.score-grid,.stp-body,.tbody,.ledgers{grid-template-columns:minmax(0,1fr)}
  .hero::before{background:rgba(255,255,255,.8)}
  .form{border-right:0;border-bottom:1px solid var(--line)}
  .ph{grid-template-columns:1fr}
  .pnum{font-size:4rem}
  .legend{grid-template-columns:1fr}
  .stp-rail button{flex-basis:76px}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}
  .js .rv{opacity:1!important;transform:none!important}
  .gfx .qa{opacity:.12}.gfx .pt{opacity:1;transform:none}.gfx .cross{stroke-dashoffset:0}.gfx .lab{opacity:1}.gfx .ring{display:none}
  .row .meter i{transform:none}.pnum{color:var(--q)}.kicker::before{transform:none}
}

/* ===== identity: paper, ink, one signal colour, hard shadows ===== */
:root{--paper:#E9E4D8;--card:#F6F2E8;--ink:#101010;--ink2:#33322E;--muted:#6B675C;--line:#CFC8B6;--tint:#EFEADD;--coral:#FF4B1F;--coral-d:#C2300F;--coral-l:#FFD9CC;
--q0:#4A4F5C;--q1:#B05A0A;--q2:#0B6E74;--q3:#A31D3A;
--serif:"Instrument Serif","Iowan Old Style",Georgia,"Times New Roman",serif;--sans:"Hanken Grotesk",Calibri,"Segoe UI",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
html,body{background:var(--paper)}
body,button,input,textarea,.pill,.btn-run,.btn-line,.chip,.ctl,.yes,.no,.legend b,.gfx text,.rail a .tip,.conv small{font-family:var(--sans)}
body{background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .07 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>")}
.mono,.tbar .u,.pnum,.num b,.ln,.lines,.rawpre{font-family:var(--mono)}
::selection{background:var(--coral);color:#fff}
.bar{height:4px;background:var(--coral)}

/* rail */
.rail{background:rgba(233,228,216,.94);border-bottom:1.5px solid var(--ink)}
.rail a{border-radius:3px;border:1.5px solid transparent;font-family:var(--mono)}
.rail a:hover{background:var(--card);border-color:var(--ink);transform:translate(-1px,-1px)}
.rail a.on{background:var(--ink);color:var(--paper)}
.rail .brand{font-family:var(--serif);font-size:1.5rem;font-weight:400;letter-spacing:0}
@media (min-width:1240px){.rail{border:1.5px solid var(--ink);border-radius:4px;background:var(--card);box-shadow:5px 5px 0 var(--ink)}}
.pill{background:var(--card);border:1.5px solid var(--ink);border-radius:3px;box-shadow:3px 3px 0 var(--ink);text-decoration:none}

/* hero */
.hero{background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);background-size:64px 64px;background-position:-1px -1px;padding-bottom:clamp(48px,8vh,88px)}
.hero::before{background:linear-gradient(90deg,var(--paper) 20%,rgba(233,228,216,.78) 58%,rgba(233,228,216,.35) 100%)}
.hero .page{grid-template-columns:minmax(0,1.08fr) minmax(0,1fr)}
.kicker{font-family:var(--mono);font-size:.78rem;color:var(--ink)}
.kicker::before{height:10px;width:10px;border-radius:50%;background:var(--coral);animation:pulse 1.8s ease-in-out infinite}
h1{font-family:var(--serif);font-weight:400;font-size:clamp(3.4rem,8.4vw,7rem);line-height:.9;letter-spacing:-.025em;margin:20px 0 22px}
h1 .u{font-style:italic;color:var(--coral)}
h1 .u::after{display:none}
.lede{font-size:clamp(1.05rem,1.7vw,1.22rem);color:var(--ink2);max-width:520px}
.stats span{background:var(--card);border:1.5px solid var(--ink);border-radius:3px;font-family:var(--mono);font-size:.84rem}
.stats b{color:var(--coral-d)}
.ticker{background:var(--ink);color:var(--paper);overflow:hidden;white-space:nowrap;border-block:1.5px solid var(--ink)}
.ticker>div{display:inline-flex;animation:tick 46s linear infinite}
.ticker:hover>div{animation-play-state:paused}
.ticker a{font:700 .92rem var(--mono);padding:13px 24px;color:var(--paper);text-decoration:none;display:inline-flex;gap:10px;align-items:center}
.ticker a i{width:8px;height:8px;border-radius:50%;background:var(--q)}
.ticker a:hover{color:var(--coral)}
@keyframes tick{to{transform:translateX(-50%)}}

/* buttons: squared, hard shadow, each kind distinct */
.btn-run{border-radius:3px;background:var(--coral);color:#fff;border:1.5px solid var(--ink);box-shadow:4px 4px 0 var(--ink);font-weight:800;letter-spacing:.01em}
.btn-run:hover{background:var(--coral);box-shadow:7px 7px 0 var(--ink);transform:translate(-3px,-3px)}
.btn-run:active{transform:translate(3px,3px);box-shadow:0 0 0 var(--ink)}
.btn-run[disabled]{background:var(--ink);box-shadow:none;transform:translate(4px,4px)}
.btn-line{border-radius:3px;background:var(--card);border:1.5px solid var(--ink);box-shadow:4px 4px 0 var(--ink)}
.btn-line::before{background:var(--coral)}
.btn-line:hover{color:#fff;transform:translate(-2px,-2px);box-shadow:6px 6px 0 var(--ink)}
.chip{background:var(--card);border:1.5px solid var(--ink);border-radius:3px;box-shadow:2px 2px 0 var(--ink)}
.chip:hover{border-color:var(--ink);color:var(--ink);background:var(--coral-l);transform:translate(-1px,-1px);box-shadow:3px 3px 0 var(--ink)}
.chip[aria-pressed=true]{background:var(--ink);color:var(--paper);box-shadow:none;transform:translate(2px,2px)}
.copy{background:var(--card);border-radius:3px}
.yes{border-radius:3px;background:var(--ink);box-shadow:3px 3px 0 var(--coral)}
.no{border-radius:3px;background:var(--card);border-color:var(--ink);color:var(--ink);box-shadow:3px 3px 0 var(--ink)}
.no::before{background:var(--coral)}
.ctl{background:var(--card);border:1.5px solid var(--ink);border-radius:3px}
.ctl.play{background:var(--card)}
.inp{background:var(--card);border:1.5px solid var(--ink);border-radius:3px}
.inp:hover{border-color:var(--ink)}
.inp:focus{box-shadow:3px 3px 0 var(--coral);border-color:var(--ink)}

/* containers */
.gcard{background:var(--card);border:1.5px solid var(--ink);border-radius:4px;box-shadow:8px 8px 0 var(--ink)}
.gfx .gl{stroke:var(--line)}
.gfx .dv{stroke:var(--ink)}
.legend button{background:var(--card);border:1.5px solid var(--ink);border-radius:3px}
.legend button:hover,.legend button.on{background:var(--coral-l);border-color:var(--ink);transform:translate(-2px,-2px);box-shadow:3px 3px 0 var(--ink)}
.legend i{border-radius:50%}
.tipbox{border-radius:3px;background:var(--ink)}
.proj{border-top:0;position:relative}
.proj::before{content:"";position:absolute;left:0;right:0;top:0;height:10px;background:var(--q)}
.proj:nth-of-type(even){background:rgba(255,255,255,.28)}
.pnum{font-family:var(--serif);font-weight:400;font-size:clamp(5rem,13vw,10rem);line-height:.8;letter-spacing:-.04em;-webkit-text-stroke:1.5px var(--q)}
.pkick{font-family:var(--mono);color:var(--ink)}
h2{font-family:var(--serif);font-weight:400;font-size:clamp(2.4rem,6vw,4.4rem);line-height:.98;letter-spacing:-.02em}
.problem{color:var(--ink2)}
.bh{font-family:var(--serif);font-weight:400;font-size:1.7rem}
.bh span{border-radius:3px;width:34px;height:34px;font-family:var(--mono)}
.bh::after{background:var(--ink);height:1.5px}
.num,.led,.card{background:var(--card);border:1.5px solid var(--ink);border-radius:4px}
.num b{font-family:var(--serif);font-weight:400;font-size:clamp(3.6rem,7vw,5.4rem);letter-spacing:-.03em;line-height:.95}
.region b{border-radius:3px}
.why{border-left-width:5px}
.conv{background:var(--paper);border-radius:3px;border:1.5px dashed var(--ink)}
.row .meter{background:var(--line);border-radius:0;height:8px}.row .meter i{border-radius:0}
.stp-rail i{background:var(--card);border-radius:4px;border:1.5px solid var(--ink)}
.stp-rail button.on i{background:var(--q);color:#fff;border-color:var(--ink);box-shadow:3px 3px 0 var(--ink);transform:translate(-2px,-2px)}
.card h4{font-family:var(--serif);font-weight:400;font-size:2rem}
.code{background:#121212;border-radius:4px;border:1.5px solid var(--ink);box-shadow:6px 6px 0 var(--q)}
.try{background:var(--card);border:1.5px solid var(--ink);border-radius:4px;box-shadow:8px 8px 0 var(--ink)}
.tbar{background:var(--paper);border-bottom:1.5px solid var(--ink)}
.tbar .m{border-radius:3px;background:var(--coral)}
.form{border-right:1.5px solid var(--ink)}
.prog{background:var(--line);height:4px}
.err{border-radius:3px;background:var(--coral-l);border:1.5px solid var(--ink);border-left:8px solid var(--coral)}
.badge{border-radius:3px;border:1.5px solid var(--ink);background:var(--card)}
.badge.g{background:#CDEBD8;border-color:var(--ink)}.badge.r{background:var(--coral-l);border-color:var(--ink)}.badge.a{background:#FFE7B0;border-color:var(--ink)}
.text,.sub,.trail span{background:var(--paper);border-radius:3px}
.trail span{background:var(--card);border-color:var(--ink)}
.took{color:var(--ink);background:#CDEBD8;border:1.5px solid var(--ink);border-radius:3px;padding:2px 10px;align-self:flex-start}
.foot{border-top:1.5px solid var(--ink)}

/* run monitor: what the waiting state shows */
.mon{display:flex;flex-direction:column;gap:16px}
.mon-top{display:flex;gap:18px;align-items:center}
.clock{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;justify-content:center;width:104px;height:104px;border:1.5px solid var(--ink);border-radius:4px;background:var(--ink);color:var(--paper)}
.clock b{font-size:2.6rem;line-height:1;letter-spacing:-.04em}
.clock span{font:700 .68rem var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--coral)}
.mon-say b{display:block;font:400 1.9rem/1.05 var(--serif)}
.mon-say span{display:block;margin-top:4px;font-size:.95rem;color:var(--ink2)}
.mon-flow{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border:1.5px solid var(--ink);border-radius:3px;overflow:hidden}
.mon-flow span{padding:8px 10px;text-align:center;font:700 .78rem var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:var(--paper);border-left:1.5px solid var(--ink)}
.mon-flow span:first-child{border-left:0}
.mon-flow span.on{background:#CDEBD8;color:var(--ink)}
.mon-flow span[data-w]{background:var(--coral);color:#fff;position:relative;overflow:hidden}
.mon-flow span[data-w]::after{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent,rgba(255,255,255,.45),transparent);transform:translateX(-100%);animation:sheen 1.6s ease-in-out infinite}
@keyframes sheen{to{transform:translateX(100%)}}
.mon-plan .k{display:block;font:700 .72rem var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.mon-plan ol{list-style:none;display:flex;flex-direction:column;gap:6px}
.mon-plan li{display:grid;grid-template-columns:28px 1fr;gap:10px;align-items:center;font-size:.95rem;animation:swap .5s var(--ease) both;animation-delay:calc(var(--k)*120ms)}
.mon-plan li i{display:grid;place-items:center;width:28px;height:28px;border:1.5px dashed var(--ink);border-radius:3px;font:700 .78rem var(--mono);font-style:normal;animation:blink 2.4s ease-in-out infinite;animation-delay:calc(var(--k)*300ms)}
@keyframes blink{50%{background:var(--coral-l)}}
.mon-plan .hint{margin-top:8px}
.skel{display:flex;flex-direction:column;gap:8px}
.skel i{display:block;height:12px;border-radius:2px;background:linear-gradient(90deg,var(--line) 25%,var(--paper) 50%,var(--line) 75%);background-size:200% 100%;animation:shim 1.4s linear infinite}
.skel i:nth-child(2){width:86%}.skel i:nth-child(3){width:62%}
@keyframes shim{to{background-position:-200% 0}}
.empty p b{display:block;color:var(--ink);font:400 1.4rem var(--serif);margin-bottom:4px}
.empty svg{stroke:var(--ink)}
@media (max-width:900px){.form{border-right:0;border-bottom:1.5px solid var(--ink)}.mon-top{flex-direction:column;align-items:flex-start}.hero .page{grid-template-columns:minmax(0,1fr)}.hero::before{background:rgba(233,228,216,.8)}}
.gfx .lab{paint-order:stroke;stroke:var(--card);stroke-width:6px;stroke-linejoin:round}

/* narrower side margins */
.page{max-width:1480px;padding:0 clamp(14px,2.6vw,36px)}
@media (min-width:1240px){.page{padding-left:104px;padding-right:40px}}
.hero .page{gap:clamp(20px,3.4vw,48px)}

/* logo */
.rail .brand{display:inline-flex;align-items:center;gap:10px;font:800 1.02rem var(--mono);letter-spacing:.06em;text-decoration:none;color:var(--ink);margin-right:10px;white-space:nowrap}
.lg{display:inline-block;line-height:0}
.lg svg{display:block;width:100%;height:100%}
.rail .brand .lg{width:34px;height:34px}
.lg.big{width:76px;height:76px;margin-bottom:6px}
@media (min-width:1240px){.rail .brand{display:none}}
.rail a.brand{min-width:0;height:auto;border:0;border-radius:0;display:inline-flex;transform:none;background:none}
.rail a.brand::after{display:none}
.rail a.brand:hover{background:none;transform:none;color:var(--coral-d)}
@media (min-width:1240px){.rail a.brand{display:none}}
"""
SITE_JS=r"""(function(){
"use strict";
var D=window.DATA,P=D.projects;
var Q=["Simple Crews","Complex Crews","Flows with direct LLM calls","Flows orchestrating Crews"];
var $=function(s,r){return (r||document).querySelector(s)};
var $$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
var reduce=window.matchMedia&&matchMedia("(prefers-reduced-motion: reduce)").matches;
function esc(t){return String(t).replace(/[&<>"]/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})}
function qi(p){return Q.indexOf(p.quadrant)}
var ARROW='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
var CP='<svg class="cp" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg><svg class="ck" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';

/* ---------- progress bar, rail, reveal ---------- */
var bar=$(".bar"),tick=false;
function onScroll(){var h=document.documentElement.scrollHeight-innerHeight;bar.style.transform="scaleX("+(h>0?Math.min(1,scrollY/h):0)+")";tick=false}
addEventListener("scroll",function(){if(!tick){tick=true;requestAnimationFrame(onScroll)}},{passive:true});onScroll();

function countUp(el){
  var to=+el.getAttribute("data-count"),t0=null,d=900;
  if(reduce){el.textContent=to;return}
  el.textContent="0";
  function f(t){if(!t0)t0=t;var p=Math.min(1,(t-t0)/d);el.textContent=Math.round(to*(1-Math.pow(1-p,3)));if(p<1)requestAnimationFrame(f)}
  requestAnimationFrame(f);
}
function reveal(e){e.classList.add("in");$$("[data-count]",e).forEach(countUp)}
var revealables=$$(".rv,.gfx");
if(!("IntersectionObserver" in window)){revealables.forEach(reveal)}
else{
  var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){reveal(e.target);io.unobserve(e.target)}})},{threshold:.18,rootMargin:"0px 0px -6% 0px"});
  revealables.forEach(function(e){io.observe(e)});
  var rail=$$(".rail a"),so=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){var n=e.target.getAttribute("data-n");rail.forEach(function(a){a.classList.toggle("on",a.getAttribute("data-n")===n);if(a.classList.contains("on")&&getComputedStyle($(".rail")).position==="sticky"){a.scrollIntoView({inline:"center",block:"nearest"})}})}})},{rootMargin:"-45% 0px -50% 0px"});
  $$(".proj").forEach(function(s){so.observe(s)});
}

/* ---------- copy ---------- */
document.addEventListener("click",function(ev){
  var b=ev.target.closest&&ev.target.closest("[data-copy]");if(!b)return;
  var src=b.getAttribute("data-copy"),text=src==="@code"?b.parentNode.getAttribute("data-text"):src;
  function done(){b.classList.add("ok");var l=$(".lb",b);if(l){l.dataset.o=l.dataset.o||l.textContent;l.textContent="Copied"}setTimeout(function(){b.classList.remove("ok");if(l)l.textContent=l.dataset.o},1600)}
  function fb(){var t=document.createElement("textarea");t.value=text;t.style.cssText="position:fixed;opacity:0";document.body.appendChild(t);t.select();try{document.execCommand("copy");done()}catch(e){}document.body.removeChild(t)}
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(text).then(done,fb)}else fb();
});

/* ---------- the graph (scores 1 to 10, split between 4 and 5, as in the notes) ---------- */
function graph(host,opts){
  var master=!!opts.master,active=opts.active,W=master?560:430,H=master?470:380,x0=46,y0=36,w=W-x0-14,h=H-y0-46;
  var X=function(v){return x0+(v-0.5)/10*w},Y=function(v){return y0+h-(v-0.5)/10*h},sx=X(4.5),sy=Y(4.5);
  var s='<svg viewBox="0 0 '+W+" "+H+'" role="img" aria-label="'+(master?"All ten projects on the complexity and precision graph":"Project "+(active&&active.n)+" on the complexity and precision graph")+'">';
  var quads=[[x0,sy,sx-x0,y0+h-sy,0],[sx,sy,x0+w-sx,y0+h-sy,1],[x0,y0,sx-x0,sy-y0,2],[sx,y0,x0+w-sx,sy-y0,3]];
  quads.forEach(function(q){
    var on=!master&&active&&qi(active)===q[4];
    s+='<rect class="qa" data-q="'+q[4]+'" x="'+q[0].toFixed(1)+'" y="'+q[1].toFixed(1)+'" width="'+q[2].toFixed(1)+'" height="'+q[3].toFixed(1)+'" fill="var(--q'+q[4]+')" style="'+(master||on?"":"display:none")+'"/>';
  });
  for(var t=1;t<=10;t++){
    s+='<line class="gl" x1="'+X(t).toFixed(1)+'" y1="'+y0+'" x2="'+X(t).toFixed(1)+'" y2="'+(y0+h)+'"/><text class="tk" x="'+X(t).toFixed(1)+'" y="'+(y0+h+14)+'" text-anchor="middle">'+t+"</text>";
    s+='<line class="gl" x1="'+x0+'" y1="'+Y(t).toFixed(1)+'" x2="'+(x0+w)+'" y2="'+Y(t).toFixed(1)+'"/><text class="tk" x="'+(x0-8)+'" y="'+(Y(t)+3).toFixed(1)+'" text-anchor="end">'+t+"</text>";
  }
  s+='<line class="dv" x1="'+sx.toFixed(1)+'" y1="'+y0+'" x2="'+sx.toFixed(1)+'" y2="'+(y0+h)+'"/><line class="dv" x1="'+x0+'" y1="'+sy.toFixed(1)+'" x2="'+(x0+w)+'" y2="'+sy.toFixed(1)+'"/>';
  var ql=[[x0+8,y0+h-8,"start",0],[x0+w-8,y0+h-8,"end",1],[x0,y0-12,"start",2],[x0+w,y0-12,"end",3]];
  ql.forEach(function(l){s+='<text class="ql'+(!master&&active&&qi(active)===l[3]?" hot":"")+'" data-q="'+l[3]+'" style="--q:var(--q'+l[3]+')" x="'+l[0]+'" y="'+l[1]+'" text-anchor="'+l[2]+'">'+Q[l[3]]+"</text>"});
  s+='<text class="ax" x="'+(x0+w/2)+'" y="'+(H-6)+'" text-anchor="middle">Complexity (1 to 10) →</text><text class="ax" transform="translate(12 '+(y0+h/2)+') rotate(-90)" text-anchor="middle">Precision (1 to 10) →</text>';
  if(!master&&active){
    var ax=X(active.c),ay=Y(active.p);
    s+='<g style="--q:var(--q'+qi(active)+')"><line class="cross" pathLength="200" x1="'+ax+'" y1="'+ay+'" x2="'+ax+'" y2="'+(y0+h)+'"/><line class="cross" pathLength="200" x1="'+ax+'" y1="'+ay+'" x2="'+x0+'" y2="'+ay+'"/></g>';
  }
  P.forEach(function(p,k){
    var me=!master&&active&&active.n===p.n,r=master?14:(me?13:9);
    s+='<g transform="translate('+X(p.c).toFixed(1)+" "+Y(p.p).toFixed(1)+')"><g class="pt'+(me?" me":"")+'" style="--i:'+k+";--q:var(--q"+qi(p)+')" tabindex="0" role="link" aria-label="Project '+p.n+": "+esc(p.title)+", complexity "+p.c+", precision "+p.p+'" data-n="'+p.n+'">'
      +(me?'<circle class="ring" r="13"/>':"")+'<circle r="'+r+'" fill="var(--q'+qi(p)+')" stroke="#fff" stroke-width="2.5"/><text text-anchor="middle" y="3.6">'+p.n+"</text></g></g>";
  });
  if(!master&&active){
    var px=X(active.c),py=Y(active.p),right=px>x0+w*.6;
    s+='<text class="lab" style="--q:var(--q'+qi(active)+')" x="'+(right?px-20:px+20)+'" y="'+(py>y0+50?py-14:py+26)+'" text-anchor="'+(right?"end":"start")+'">Complexity '+active.c+" · Precision "+active.p+"</text>";
  }
  s+="</svg>";
  host.innerHTML=s;
  if(!master)host.classList.add("faint");
  var tip=opts.tip;
  $$(".pt",host).forEach(function(g){
    var n=+g.getAttribute("data-n"),p=P[n-1];
    function go(){var t=$("#p"+n);if(t)t.scrollIntoView({behavior:reduce?"auto":"smooth"})}
    g.addEventListener("click",go);g.addEventListener("keydown",function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();go()}});
    if(tip){
      var show=function(){var b=host.getBoundingClientRect(),r=g.getBoundingClientRect(),pr=host.parentNode.getBoundingClientRect();tip.innerHTML="<b>"+String(n).padStart(2,"0")+" "+esc(p.title)+"</b><span>Complexity "+p.c+", precision "+p.p+"<br>"+p.quadrant+"</span>";tip.style.left=Math.min(pr.width-230,Math.max(8,r.left-pr.left+r.width/2-100))+"px";tip.style.top=(r.top-pr.top-tip.offsetHeight-12)+"px";tip.classList.add("on")};
      g.addEventListener("mouseenter",show);g.addEventListener("focus",show);g.addEventListener("mouseleave",function(){tip.classList.remove("on")});g.addEventListener("blur",function(){tip.classList.remove("on")});
    }
  });
}
var master=$("#master");
if(master){
  graph(master,{master:true,tip:$("#tip")});
  $$(".legend button").forEach(function(b){
    var q=+b.getAttribute("data-q");
    function on(){b.classList.add("on");$$("#master .pt").forEach(function(g){var p=P[+g.getAttribute("data-n")-1];g.style.opacity=qi(p)===q?1:.22});$$("#master .ql").forEach(function(l){l.classList.toggle("hot",+l.getAttribute("data-q")===q)})}
    function off(){b.classList.remove("on");$$("#master .pt").forEach(function(g){g.style.opacity=""});$$("#master .ql").forEach(function(l){l.classList.remove("hot")})}
    b.addEventListener("mouseenter",on);b.addEventListener("focus",on);b.addEventListener("mouseleave",off);b.addEventListener("blur",off);
    b.addEventListener("click",function(){var p=P.filter(function(p){return qi(p)===q})[0];if(p)$("#p"+p.n).scrollIntoView({behavior:reduce?"auto":"smooth"})});
  });
}
$$(".gfx[data-n]").forEach(function(h){graph(h,{active:P[+h.getAttribute("data-n")-1]})});

/* ---------- Python highlighting, one line at a time ---------- */
var KW="and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield".split(" ");
var TQ="\u0022\u0022\u0022";
var TOK=/(#[^\n]*)|(f?\x22{3}|f?"(?:\\.|[^"\\])*"|f?'(?:\\.|[^'\\])*')|(@[A-Za-z_][\w.]*)|\b([A-Za-z_]\w*)\b|(\b\d+(?:\.\d+)?\b)/g;
function hl(line,st){
  var out="",last=0,m,prev="";
  if(st.tri){
    var e=line.indexOf(TQ);
    if(e<0)return '<span class="ts">'+esc(line)+"</span>";
    out='<span class="ts">'+esc(line.slice(0,e+3))+"</span>";last=e+3;st.tri=false;
  }
  TOK.lastIndex=last;
  while((m=TOK.exec(line))){
    out+=esc(line.slice(last,m.index));last=TOK.lastIndex;
    if(m[1])out+='<span class="tc">'+esc(m[1])+"</span>";
    else if(m[2]){
      if(/\x22{3}$/.test(m[2])&&m[2].length<=4){var c=line.indexOf(TQ,last);if(c<0){out+='<span class="ts">'+esc(m[2]+line.slice(last))+"</span>";last=line.length;st.tri=true;TOK.lastIndex=last}else{out+='<span class="ts">'+esc(m[2]+line.slice(last,c+3))+"</span>";last=c+3;TOK.lastIndex=last}}
      else out+='<span class="ts">'+esc(m[2])+"</span>";
    }
    else if(m[3])out+='<span class="td">'+esc(m[3])+"</span>";
    else if(m[4]){var w=m[4];if(KW.indexOf(w)>-1)out+='<span class="tk2">'+w+"</span>";else if(prev==="def"||prev==="class"||line.charAt(last)==="(")out+='<span class="tf">'+w+"</span>";else out+=w;prev=w;continue}
    else if(m[5])out+='<span class="tn">'+m[5]+"</span>";
    prev="";
  }
  return out+esc(line.slice(last));
}

/* ---------- the seven steps ---------- */
function stepper(host,p){
  var q="var(--q"+qi(p)+")",cur=0,timer=null,html='<div class="stp-rail" role="tablist" aria-label="The seven steps of project '+p.n+'">';
  p.steps.forEach(function(s,i){html+='<button type="button" role="tab" id="t'+p.n+"_"+i+'" aria-selected="false" data-i="'+i+'"><i>'+s.n+"</i><span>"+esc(s.name)+"</span></button>"});
  html+='</div><div class="stp-body"><div class="card" aria-live="polite"></div><div class="code" data-text=""><button type="button" class="copy" data-copy="@code">'+CP+'<span class="lb">Copy code</span></button><div class="cap"><i></i><i></i><i></i><span>app.py · project '+p.n+'</span></div><div class="lines" tabindex="0" aria-label="Code of project '+p.n+'"></div></div></div>';
  host.innerHTML=html;
  var rail=$(".stp-rail",host),card=$(".card",host),lines=$(".lines",host),codeBox=$(".code",host);
  codeBox.setAttribute("data-text",p.code.join("\n"));
  var st={tri:false},rows="";
  p.code.forEach(function(l,i){rows+='<div class="ln" data-i="'+i+'"><s>'+(i+1)+"</s><span>"+hl(l,st)+"</span></div>"});
  lines.innerHTML=rows;
  var ln=$$(".ln",lines);
  function range(n){return p.ranges.filter(function(r){return r.n===n})[0]}
  function go(i,user){
    cur=i;var s=p.steps[i],r=range(s.n);
    $$("button",rail).forEach(function(b,k){b.classList.toggle("on",k===i);b.classList.toggle("done",k<i);b.setAttribute("aria-selected",k===i?"true":"false");b.tabIndex=k===i?0:-1});
    ln.forEach(function(el,k){var on=r&&k>=r.start&&k<=r.end;el.classList.toggle("on",!!on);el.classList.toggle("first",!!(r&&k===r.start))});
    card.innerHTML='<div class="swap"><span class="from">From your notes</span><h4>Step '+s.n+" · <em>"+esc(s.name)+'</em></h4><p class="note">'+esc(s.note)+"</p>"+(s.here?'<p class="here"><b>In this project.</b> '+esc(s.here)+"</p>":"")+(r?'<p class="hint">Lines '+(r.start+1)+" to "+(r.end+1)+" of the code.</p>":"")+'</div><div class="ctrls"><button type="button" class="ctl back"'+(i===0?" disabled":"")+'>← Back</button><button type="button" class="ctl next"'+(i===6?" disabled":"")+'>Next →</button><button type="button" class="ctl play" aria-pressed="'+(!!timer)+'"><svg viewBox="0 0 24 24"><path d="M7 5l12 7-12 7z"/></svg><span>'+(timer?"Pause":"Walk me through")+"</span></button></div>";
    if(r){var top=ln[r.start].offsetTop-lines.offsetTop-24;lines.scrollTo({top:Math.max(0,top),behavior:reduce||!user&&!timer&&cur===0?"auto":"smooth"})}
    $(".back",card).onclick=function(){stop();go(cur-1,true)};$(".next",card).onclick=function(){stop();go(cur+1,true)};
    $(".play",card).onclick=function(){timer?stop(true):play()};
  }
  function stop(redraw){if(timer){clearInterval(timer);timer=null;if(redraw)go(cur,true)}}
  function play(){var first=cur>=6?0:cur+1;timer=setInterval(function(){if(cur>=6){stop(true);return}go(cur+1,true)},3800);go(first,true)}
  rail.addEventListener("click",function(e){var b=e.target.closest("button");if(b){stop();go(+b.getAttribute("data-i"),true)}});
  rail.addEventListener("keydown",function(e){var d=e.key==="ArrowRight"?1:e.key==="ArrowLeft"?-1:0;if(d){e.preventDefault();var n=Math.max(0,Math.min(6,cur+d));stop();go(n,true);$("#t"+p.n+"_"+n).focus()}});
  go(0,false);
}
$$(".stepper").forEach(function(h){stepper(h,P[+h.getAttribute("data-n")-1])});

/* ---------- run it ---------- */
function label(k){k=k.replace(/_/g," ");return k.charAt(0).toUpperCase()+k.slice(1)}
function badge(s){var c="";if(/^(shortlist|hot|published)$/.test(s))c="g";else if(/^(reject|cold|critical)$/.test(s))c="r";else if(/^(warm|high|awaiting_review|revise)$/.test(s))c="a";return '<span class="badge '+c+'">'+esc(s.replace(/_/g," "))+"</span>"}
function md(t){return esc(t).replace(/^#{1,4}\s*(.+)$/gm,"<h5>$1</h5>").replace(/\*\*(.+?)\*\*/g,"<b>$1</b>")}
function val(v){
  if(v===null||v===undefined||v==="")return '<span class="hint">none</span>';
  if(typeof v==="boolean")return '<span class="badge '+(v?"g":"r")+'">'+(v?"yes":"no")+"</span>";
  if(typeof v==="number")return '<b class="mono">'+v+"</b>";
  if(typeof v==="string"){if(v.length<=26&&/^[a-z_]+$/.test(v))return badge(v);if(v.length>110||v.indexOf("\n")>-1)return '<div class="text">'+md(v)+"</div>";return esc(v)}
  if(Array.isArray(v)){if(!v.length)return '<span class="hint">none</span>';if(typeof v[0]!=="object")return '<ul class="list">'+v.map(function(i){return "<li>"+esc(i)+"</li>"}).join("")+"</ul>";return v.map(function(o){return '<div class="sub">'+obj(o)+"</div>"}).join("")}
  return obj(v);
}
function obj(o){return Object.keys(o).filter(function(k){return k!=="trail"}).map(function(k){return '<div class="kv"><span class="k">'+esc(label(k))+'</span><div class="v">'+val(o[k])+"</div></div>"}).join("")}
function errText(r,j){var d=j&&j.detail;if(Array.isArray(d))return d.map(function(e){return (e.loc||[]).slice(1).join(".")+": "+e.msg}).join(" | ");if(typeof d==="string")return d;return "The server answered with status "+r.status+"."}

function tryPanel(host,p){
  var n=p.n,state={draft:"",busy:false,t0:0,tick:null},reviewing=p.reviewing;
  host.innerHTML='<div class="tbar"><span class="m">POST</span><span class="u mono">'+esc(location.origin)+"/agent"+n+'</span><button type="button" class="copy" data-copy="'+esc(location.origin)+"/agent"+n+'">'+CP+'<span class="lb">Copy link</span></button><a href="/agent'+n+'" target="_blank" rel="noopener">GET description</a><a href="/agent'+n+'?run=true" target="_blank" rel="noopener">Run in browser</a></div><div class="prog"><i></i></div>'
    +'<div class="tbody"><form class="form" novalidate><div class="fld"><span class="flab">Start from a sample</span><div class="chips"></div></div><div class="act"><button class="btn-run" type="submit"><span class="sp"></span><span class="lb">Run agent '+n+"</span>"+ARROW+'</button><span class="hint">'+(reviewing?"The first run returns a draft. You then approve it or request changes, which is the human step of this Flow.":"A run makes several model calls and can take a minute or more.")+'</span></div></form>'
    +'<div class="out" aria-live="polite"><div class="empty"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M7 9h10M7 13h6"/></svg><p><b>Nothing has run yet.</b> Choose a sample, then press Run. The answer, and the path the agents took, appears here.</p></div></div></div>';
  var form=$(".form",host),chips=$(".chips",host),out=$(".out",host),run=$(".btn-run",host),act=$(".act",host);
  act.insertAdjacentHTML("beforebegin",p.fields.map(function(f,i){
    var id="f"+n+"_"+i,h='<div class="fld"><label for="'+id+'">'+esc(f.label)+"</label>";
    h+=f.kind==="area"?'<textarea class="inp" id="'+id+'" name="'+f.name+'" rows="'+(f.rows||5)+'" spellcheck="false"></textarea>':'<input class="inp" id="'+id+'" name="'+f.name+'" type="text" autocomplete="off">';
    return h+(f.hint?'<span class="hint">'+esc(f.hint)+"</span>":"")+"</div>";
  }).join(""));
  function fill(s,btn){p.fields.forEach(function(f){var el=form.elements[f.name];if(el)el.value=s.values[f.name]==null?"":s.values[f.name]});$$("button",chips).forEach(function(b){b.setAttribute("aria-pressed",b===btn?"true":"false")});state.draft=""}
  p.samples.forEach(function(s,i){var b=document.createElement("button");b.type="button";b.className="chip";b.textContent=s.label;b.setAttribute("aria-pressed","false");b.addEventListener("click",function(){fill(s,b)});chips.appendChild(b)});
  fill(p.samples[0],chips.firstElementChild);
  function collect(){var o={};p.fields.forEach(function(f){var v=form.elements[f.name].value;
    if(f.as==="urls")o[f.name]=v.split(/\n+/).map(function(s){return s.trim()}).filter(Boolean);
    else if(f.as==="resumes")o[f.name]=v.split(/\n\s*={3,}\s*\n/).map(function(b){b=b.trim();var k=b.indexOf("\n");return k<0?null:{name:b.slice(0,k).trim(),text:b.slice(k+1).trim()}}).filter(Boolean);
    else if(f.optional&&!v.trim())return;else o[f.name]=v});return o}
  function monitor(){
    var steps=p.plan.map(function(s,k){return '<li style="--k:'+k+'"><i>'+(k+1)+"</i><span>"+esc(s)+"</span></li>"}).join("");
    return '<div class="mon"><div class="mon-top"><div class="clock"><b class="mono" data-clock>0</b><span>seconds</span></div><div class="mon-say"><b data-say>Request sent</b><span data-sub>The server has your input and is calling Gemini.</span></div></div>'
      +'<div class="mon-flow"><span class="on">Sent</span><span class="on" data-w>Agents working</span><span>Result</span></div>'
      +'<div class="mon-plan"><span class="k">This run will pass through</span><ol>'+steps+'</ol><p class="hint">The server reports the exact path when it finishes. Until then this list shows the expected order, not live progress.</p></div>'
      +'<div class="skel"><i></i><i></i><i></i></div></div>';
  }
  function busy(on){state.busy=on;host.classList.toggle("busy",on);run.disabled=on;$$(".yes,.no",host).forEach(function(b){b.disabled=on});var l=$(".lb",run);
    if(on){state.t0=Date.now();l.textContent="Running 0s";
      state.tick=setInterval(function(){var s=Math.round((Date.now()-state.t0)/1000);l.textContent="Running "+s+"s";
        var c=$("[data-clock]",out);if(c)c.textContent=s;
        var say=$("[data-say]",out),sub=$("[data-sub]",out);if(!say)return;
        if(s>=75){say.textContent="Still working";sub.textContent="Longer runs are normal on the free plan. If the service had been idle it needed up to a minute to wake first. Please keep this page open."}
        else if(s>=25){say.textContent="Agents are working";sub.textContent="Each agent makes its own model call and passes its result to the next. Runs of 30 to 120 seconds are normal."}
        else if(s>=6){say.textContent="Waiting for Gemini";sub.textContent="The first call is in progress."}
      },1000)}
    else{clearInterval(state.tick);l.textContent=reviewing&&state.draft?"Start again":"Run agent "+n}}
  function post(body){
    busy(true);out.innerHTML=monitor();out.scrollIntoView&&window.innerWidth<900&&out.scrollIntoView({behavior:"smooth",block:"nearest"});
    var ctl=new AbortController(),to=setTimeout(function(){ctl.abort()},420000),t0=Date.now();
    fetch("/agent"+n,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal:ctl.signal}).then(function(r){return r.json().catch(function(){return null}).then(function(j){return{r:r,j:j}})}).then(function(x){
      clearTimeout(to);busy(false);
      if(!x.r.ok){out.innerHTML='<div class="err"><b>The request was not accepted ('+x.r.status+").</b>"+esc(errText(x.r,x.j))+(x.r.status===503||x.r.status===502||x.r.status===429?" Nothing is wrong with your input. Wait a moment and run it again.":"")+"</div>";return}
      render(x.j,Math.round((Date.now()-t0)/1000));
    }).catch(function(e){clearTimeout(to);busy(false);out.innerHTML='<div class="err"><b>No response.</b>'+(e&&e.name==="AbortError"?"The run took longer than seven minutes and was stopped.":"The browser could not reach the server. On a free plan it may be waking; try again in a minute.")+"</div>"});
  }
  function render(j,secs){
    var h='<div class="res"><span class="took">Finished in '+secs+" s</span>";
    if(j.trail&&j.trail.length){h+='<div class="kv"><span class="k">Path taken</span><div class="trail" style="--q:var(--q'+qi(p)+')">'+j.trail.map(function(t,k){return (k?'<em style="--k:'+k+'">→</em>':"")+'<span style="--k:'+k+'">'+esc(t)+"</span>"}).join("")+"</div></div>"}
    h+=obj(j)+"</div>";
    if(reviewing&&j.status==="awaiting_review"){state.draft=j.draft;h+='<div class="review"><span class="flab">Your decision</span><div class="fld"><label for="fb'+n+'">Comments (required to request changes)</label><textarea class="inp" id="fb'+n+'" rows="3" placeholder="For example: shorten the introduction and add a heading for risks."></textarea></div><div class="r2"><button type="button" class="yes" data-d="approved">Approve and publish</button><button type="button" class="no" data-d="revise">Request changes</button></div></div>'}
    else if(reviewing){state.draft=""}
    h+='<button type="button" class="raw">Show raw JSON</button><pre class="rawpre">'+esc(JSON.stringify(j,null,2))+"</pre>";
    out.innerHTML=h;
    var rb=$(".raw",out),rp=$(".rawpre",out);rb.addEventListener("click",function(){var o=rp.classList.toggle("open");rb.textContent=o?"Hide raw JSON":"Show raw JSON"});
    $$("[data-d]",out).forEach(function(b){b.addEventListener("click",function(){var d=b.getAttribute("data-d"),fb=$("#fb"+n),body=collect();if(d==="revise"&&!fb.value.trim()){fb.focus();fb.style.borderColor="#FF5A50";return}body.draft=state.draft;body.decision=d;body.feedback=fb.value.trim();post(body)})});
  }
  form.addEventListener("submit",function(e){e.preventDefault();if(state.busy)return;var miss=p.fields.filter(function(f){return !f.optional&&!String(form.elements[f.name].value).trim()});if(miss.length){var el=form.elements[miss[0].name];el.focus();el.style.borderColor="#FF5A50";setTimeout(function(){el.style.borderColor=""},1800);return}state.draft="";post(collect())});
}
$$(".try").forEach(function(h){tryPanel(h,P[+h.getAttribute("data-n")-1])});

/* ---------- server status ---------- */
var pill=$(".pill");
function status(cls,text){pill.className="pill "+cls;$(".t",pill).textContent=text}
status("wait","Checking server");
var ctl=new AbortController(),to=setTimeout(function(){ctl.abort()},90000);
fetch("/health",{signal:ctl.signal}).then(function(r){return r.json()}).then(function(j){clearTimeout(to);j.has_key?status("ok","Server ready"):status("bad","Gemini key missing")}).catch(function(){status("bad","Server not reachable")});
})();
"""
SITE_HTML=r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CREWAI: ten agents, one server</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;700;800&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;700&display=swap">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%20240%20240%22%20width=%22240%22%20height=%22240%22%20role=%22img%22%20aria-label=%22CREWAI%20icon%22%3E%3Ctitle%3ECREWAI%3C/title%3E%3Crect%20width=%22240%22%20height=%22240%22%20rx=%2244%22%20fill=%22%23E9E4D8%22/%3E%3Crect%20x=%223%22%20y=%223%22%20width=%22234%22%20height=%22234%22%20rx=%2241%22%20fill=%22none%22%20stroke=%22%23101010%22%20stroke-width=%224%22/%3E%3Cpath%20d=%22M184.09%20157.00%20A74%2074%200%201%201%20184.09%2083.00%22%20fill=%22none%22%20stroke=%22%23101010%22%20stroke-width=%222.02%22%20stroke-linecap=%22round%22/%3E%3Cline%20x1=%22184.09%22%20y1=%2283.00%22%20x2=%22132.88%22%20y2=%22120.00%22%20stroke=%22%23101010%22%20stroke-width=%222.02%22%20stroke-linecap=%22round%22/%3E%3Ccircle%20cx=%22184.09%22%20cy=%22157.00%22%20r=%226.44%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%22153.21%22%20cy=%22186.13%22%20r=%227.01%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%22111.41%22%20cy=%22193.50%22%20r=%227.58%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%2272.43%22%20cy=%22176.69%22%20r=%228.15%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%2249.11%22%20cy=%22141.22%22%20r=%228.72%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%2249.11%22%20cy=%2298.78%22%20r=%229.29%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%2272.43%22%20cy=%2263.31%22%20r=%229.86%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%22111.41%22%20cy=%2246.50%22%20r=%2210.43%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%22153.21%22%20cy=%2253.87%22%20r=%2211.00%22%20fill=%22%23101010%22/%3E%3Ccircle%20cx=%22184.09%22%20cy=%2283.00%22%20r=%2211.57%22%20fill=%22%23FF4B1F%22/%3E%3Crect%20x=%22104.36%22%20y=%22104.36%22%20width=%2231.28%22%20height=%2231.28%22%20rx=%224.60%22%20fill=%22%23101010%22/%3E%3C/svg%3E">
<meta name="description" content="Ten CrewAI agents behind one FastAPI server. Each project is scored on complexity and precision, explained in seven steps, and can be run on this page.">
<script>document.documentElement.className+=" js"</script>
<style>/*CSS*/</style></head><body>
<div class="bar" aria-hidden="true" id="top"></div>
/*RAIL*/
/*HERO*/
<main>/*SECTIONS*/</main>
/*FOOT*/
<a class="pill wait" href="/health"><i></i><span class="t">Checking server</span></a>
<noscript><p style="padding:16px">The graphs, code walk-through and run panels need JavaScript. The text above works without it, and every agent is available at /agent1 to /agent10.</p></noscript>
<script>window.DATA=/*DATA*/;</script>
<script>/*JS*/</script>
</body></html>"""
