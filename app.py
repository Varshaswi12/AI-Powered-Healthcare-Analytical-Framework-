import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request, send_file
import sqlite3
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise Exception("❌ GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

model = joblib.load(os.path.join(BASE_DIR,"model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR,"encoders.pkl"))

DISPLAY_RESULT=None
DOWNLOAD_RESULT=None


# ================= DATABASE =================
def run_query(query):
    conn=sqlite3.connect(os.path.join(BASE_DIR,"database.db"))
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df


# ================= SQL FROM LLM =================
def llm_generate_sql(user_query):

    schema = """
Table healthcare(
age INTEGER,
gender TEXT,
medical_condition TEXT,
admission_type TEXT,
test_results TEXT,
hospital TEXT,
doctor TEXT,
date_of_admission TEXT,
discharge_date TEXT
)
"""

    system_prompt = f"""
You are an expert SQL generator.

Convert user questions into VALID SQLite SQL.

STRICT RULES:
- Use ONLY table healthcare
- Always include ALL filters mentioned
- Text comparisons must use LOWER(column)
- Use = for exact matches
- Age conditions use > < >= <=
- "count" or "how many" → SELECT COUNT(*)
- "average age" → SELECT AVG(age)
- "percentage abnormal" → COUNT(*)*100.0/(SELECT COUNT(*) FROM healthcare)
- NEVER explain anything
- RETURN ONLY SQL
- SQL MUST start with SELECT

### EXAMPLES:

female diabetes patients above 40
SELECT * FROM healthcare
WHERE LOWER(gender)='female'
AND LOWER(medical_condition)='diabetes'
AND age>40;

male cancer patients below 30
SELECT * FROM healthcare
WHERE LOWER(gender)='male'
AND LOWER(medical_condition)='cancer'
AND age<30;

patients with asthma
SELECT * FROM healthcare
WHERE LOWER(medical_condition)='asthma';

female diabetes patients above 40 with abnormal test results
SELECT * FROM healthcare
WHERE LOWER(gender)='female'
AND LOWER(medical_condition)='diabetes'
AND age>40
AND LOWER(test_results)='abnormal';

count diabetes patients
SELECT COUNT(*) FROM healthcare
WHERE LOWER(medical_condition)='diabetes';

average age of cancer patients
SELECT AVG(age) FROM healthcare
WHERE LOWER(medical_condition)='cancer';

how many female patients
SELECT COUNT(*) FROM healthcare
WHERE LOWER(gender)='female';

percentage of abnormal test results
SELECT COUNT(*)*100.0/(SELECT COUNT(*) FROM healthcare)
FROM healthcare
WHERE LOWER(test_results)='abnormal';

Schema:
{schema}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_query}
        ]
    )

    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql","").replace("```","").strip()

    print("\nGENERATED SQL:", sql)


    if not sql.lower().startswith("select"):
        return "SELECT * FROM healthcare LIMIT 20"

    return sql
# ================= GLOBAL CHARTS =================
def global_analytics():

    conn=sqlite3.connect(os.path.join(BASE_DIR,"database.db"))

    g=pd.read_sql("SELECT LOWER(gender) gender, COUNT(*) c FROM healthcare GROUP BY LOWER(gender)",conn)
    c=pd.read_sql("SELECT medical_condition, COUNT(*) c FROM healthcare GROUP BY medical_condition ORDER BY c DESC LIMIT 6",conn)
    a=pd.read_sql("SELECT admission_type, COUNT(*) c FROM healthcare GROUP BY admission_type",conn)

    conn.close()

    charts={}

    if len(g)>1:
        charts["gender"]={"labels":g["gender"].tolist(),"values":g["c"].tolist()}

    if len(c)>1:
        charts["condition"]={"labels":c["medical_condition"].tolist(),"values":c["c"].tolist()}

    if len(a)>1:
        charts["admission"]={"labels":a["admission_type"].tolist(),"values":a["c"].tolist()}

    return charts


# ================= DATASET INSIGHTS =================
def dataset_insights():

    conn=sqlite3.connect(os.path.join(BASE_DIR,"database.db"))
    df=pd.read_sql("SELECT * FROM healthcare",conn)
    conn.close()

    return{
        "rows":len(df),
        "cols":len(df.columns),
        "missing":int(df.isna().sum().sum()),
        "avg_age":round(df["age"].mean(),1)
    }


# ================= KPI =================
def analytics_summary():

    conn=sqlite3.connect(os.path.join(BASE_DIR,"database.db"))

    total=pd.read_sql("SELECT COUNT(*) c FROM healthcare",conn)["c"][0]
    avg_age=pd.read_sql("SELECT AVG(age) a FROM healthcare",conn)["a"][0]

    top_condition=pd.read_sql("""
        SELECT medical_condition
        FROM healthcare
        GROUP BY medical_condition
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """,conn)["medical_condition"].iloc[0]

    abnormal=pd.read_sql("SELECT COUNT(*) c FROM healthcare WHERE LOWER(test_results)='abnormal'",conn)["c"][0]

    conn.close()

    return{
        "total":int(total),
        "avg_age":round(avg_age,1),
        "top_condition":top_condition,
        "abnormal_pct":round((abnormal/total)*100,2) if total else 0
    }


# ================= MAIN =================
@app.route("/",methods=["GET","POST"])
def index():

    global DISPLAY_RESULT,DOWNLOAD_RESULT

    sql_used=None
    analytics=analytics_summary()
    insights=dataset_insights()
    global_charts=global_analytics()
    

    if request.method=="POST":

        user_input=request.form["query"]
        sql_used=llm_generate_sql(user_input)

        try:
            full_result=run_query(sql_used)

            DOWNLOAD_RESULT=full_result
            DISPLAY_RESULT=full_result.head(20)

           

        except Exception as e:
            DISPLAY_RESULT=f"Error: {e}"

    return render_template(
        "index.html",
        tables=DISPLAY_RESULT,
        sql=sql_used,
        analytics=analytics,
        insights=insights,
        global_charts=global_charts,
        
    )


# ================= DOWNLOAD FULL =================
@app.route("/download")
def download():

    global DOWNLOAD_RESULT

    if DOWNLOAD_RESULT is None or DOWNLOAD_RESULT.empty:
        return "Run a query first."

    path=os.path.join(BASE_DIR,"static","full_result.csv")
    DOWNLOAD_RESULT.to_csv(path,index=False)

    return send_file(path,as_attachment=True)


if __name__=="__main__":
    app.run(debug=True)