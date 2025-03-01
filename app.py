import streamlit as st
import sqlite3
import pandas as pd
import json
import requests
import datetime
import os
from pathlib import Path

# Page configuration
st.set_page_config(page_title="منصة المبادرات الذكية", layout="wide", page_icon="💡")

# Database setup
def init_db():
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    conn = sqlite3.connect('data/initiatives.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS initiatives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        employee_name TEXT NOT NULL,
        department TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        goals TEXT NOT NULL,
        requirements TEXT NOT NULL,
        budget REAL,
        status TEXT DEFAULT 'pending',
        ai_feedback TEXT,
        admin_feedback TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create a table for storing historical initiatives for RAG
    c.execute('''
    CREATE TABLE IF NOT EXISTS rag_knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        category TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn

# Initialize database
conn = init_db()

# Function to call Deepseek API
def call_deepseek_api(prompt, api_key):
    url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

# Get historical initiatives for RAG context
def get_rag_context():
    df = pd.read_sql_query("SELECT content FROM rag_knowledge", conn)
    if df.empty:
        return ""
    
    context = "\n\n".join(df['content'].tolist())
    if len(context) > 10000:  # Limit context size
        context = context[:10000]
    return context

# Get AI feedback on initiative
def get_ai_feedback(initiative_data, api_key):
    rag_context = get_rag_context()
    
    prompt = f"""
    أنت مستشار ذكي لتقييم وتحسين المبادرات المقدمة من الموظفين في جهة حكومية. 
    
    فيما يلي معلومات عن مبادرات سابقة للاستفادة منها:
    {rag_context}
    
    فيما يلي تفاصيل المبادرة الجديدة المقدمة:
    
    عنوان المبادرة: {initiative_data['title']}
    القسم: {initiative_data['department']}
    الوصف: {initiative_data['description']}
    الأهداف: {initiative_data['goals']}
    المتطلبات: {initiative_data['requirements']}
    الميزانية المقترحة: {initiative_data['budget']} ريال
    
    قم بتقديم تقييم وملاحظات على المبادرة متضمناً:
    1. تقييم عام للمبادرة (قوتها، وضوحها، توافقها مع أهداف المؤسسات الحكومية)
    2. اقتراحات لتحسين المبادرة
    3. أفكار إضافية يمكن دمجها مع المبادرة
    4. تقييم واقعية الميزانية المقترحة
    5. تصنيف المبادرة (ابتكارية، تحسينية، توفيرية) 
    
    قدم إجابتك باللغة العربية بتنسيق واضح ومرتب.
    """
    
    return call_deepseek_api(prompt, api_key)

# Add a new initiative to RAG knowledge base
def add_to_rag_knowledge(initiative_data):
    cursor = conn.cursor()
    content = f"""
    عنوان المبادرة: {initiative_data['title']}
    القسم: {initiative_data['department']}
    الوصف: {initiative_data['description']}
    الأهداف: {initiative_data['goals']}
    المتطلبات: {initiative_data['requirements']}
    الميزانية: {initiative_data['budget']} ريال
    """
    
    cursor.execute(
        "INSERT INTO rag_knowledge (content, category) VALUES (?, ?)",
        (content, initiative_data['department'])
    )
    conn.commit()

# Save initiative to database
def save_initiative(initiative_data, ai_feedback):
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO initiatives 
           (employee_id, employee_name, department, title, description, goals, requirements, budget, ai_feedback) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            initiative_data['employee_id'],
            initiative_data['employee_name'],
            initiative_data['department'],
            initiative_data['title'],
            initiative_data['description'],
            initiative_data['goals'],
            initiative_data['requirements'],
            initiative_data['budget'],
            ai_feedback
        )
    )
    conn.commit()
    return cursor.lastrowid

# Get all initiatives
def get_all_initiatives():
    return pd.read_sql_query("SELECT * FROM initiatives ORDER BY created_at DESC", conn)

# Update initiative status
def update_initiative_status(initiative_id, status, admin_feedback=""):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE initiatives SET status = ?, admin_feedback = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, admin_feedback, initiative_id)
    )
    conn.commit()

# Get single initiative by ID
def get_initiative_by_id(initiative_id):
    return pd.read_sql_query("SELECT * FROM initiatives WHERE id = ?", conn, params=(initiative_id,))

# Function to ensure API key is available
def ensure_api_key():
    if "api_key" not in st.session_state or not st.session_state.api_key:
        try:
            # Try to get API key from secrets
            api_key = st.secrets["DEEPSEEK_API_KEY"]
            st.session_state.api_key = api_key
        except Exception:
            # Fall back to requesting from user
            st.sidebar.header("إعدادات API")
            api_key = st.sidebar.text_input("أدخل مفتاح API للذكاء الاصطناعي:", type="password")
            
            if st.sidebar.button("حفظ المفتاح"):
                st.session_state.api_key = api_key
                st.sidebar.success("تم حفظ المفتاح بنجاح!")
                st.experimental_rerun()
            
            if not api_key:
                st.warning("يرجى إدخال مفتاح API للاستمرار")
                st.stop()
            
            st.session_state.api_key = api_key

# UI function for multi-page navigation
def navigation():
    with st.sidebar:
        st.title("منصة المبادرات الذكية")
        
        # Role selection
        role = st.radio(
            "اختر دورك:",
            ["موظف", "مدير", "قسم الموارد البشرية", "القسم المالي"]
        )
        
        # Navigation for employee role
        if role == "موظف":
            page = st.radio(
                "الصفحات:",
                ["تقديم مبادرة جديدة", "عرض مبادراتي"]
            )
            if page == "تقديم مبادرة جديدة":
                return "submit_initiative", role
            else:
                return "view_my_initiatives", role
        
        # Navigation for admin/HR/Finance
        else:
            return "review_initiatives", role

# UI for submitting a new initiative
def submit_initiative_page():
    st.title("📝 تقديم مبادرة جديدة")
    
    with st.form("initiative_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            employee_id = st.text_input("الرقم الوظيفي")
            employee_name = st.text_input("اسم الموظف")
            department = st.selectbox(
                "القسم",
                ["تقنية المعلومات", "الموارد البشرية", "المالية", "الخدمات", "التطوير", "أخرى"]
            )
            title = st.text_input("عنوان المبادرة")
        
        with col2:
            budget = st.number_input("الميزانية المقترحة (ريال)", min_value=0.0, step=1000.0)
        
        description = st.text_area("وصف المبادرة", height=150)
        goals = st.text_area("أهداف المبادرة وفوائدها", height=100)
        requirements = st.text_area("متطلبات تنفيذ المبادرة", height=100)
        
        submit_button = st.form_submit_button("تقديم المبادرة")
        
        if submit_button:
            if not employee_id or not employee_name or not title or not description or not goals:
                st.error("يرجى تعبئة جميع الحقول الإلزامية")
            else:
                # Show spinner while processing
                with st.spinner("جاري تحليل المبادرة..."):
                    # Prepare initiative data
                    initiative_data = {
                        "employee_id": employee_id,
                        "employee_name": employee_name,
                        "department": department,
                        "title": title,
                        "description": description,
                        "goals": goals,
                        "requirements": requirements,
                        "budget": budget
                    }
                    
                    # Get AI feedback
                    ai_feedback = get_ai_feedback(initiative_data, st.session_state.api_key)
                    
                    # Save initiative to database
                    initiative_id = save_initiative(initiative_data, ai_feedback)
                    
                    # Add to RAG knowledge base
                    add_to_rag_knowledge(initiative_data)
                    
                    st.success(f"تم تقديم المبادرة بنجاح! رقم المبادرة: {initiative_id}")
                
                # Display AI feedback
                st.subheader("تحليل المبادرة بواسطة الذكاء الاصطناعي")
                st.write(ai_feedback)

# UI for viewing employee's initiatives
def view_my_initiatives_page(employee_id=""):
    st.title("🔍 عرض مبادراتي")
    
    if not employee_id:
        employee_id = st.text_input("أدخل الرقم الوظيفي")
        if st.button("بحث"):
            pass
        else:
            return
    
    initiatives = pd.read_sql_query(
        "SELECT * FROM initiatives WHERE employee_id = ? ORDER BY created_at DESC", 
        conn, 
        params=(employee_id,)
    )
    
    if initiatives.empty:
        st.info("لا توجد مبادرات مقدمة بعد.")
        return
    
    for _, initiative in initiatives.iterrows():
        with st.expander(f"**{initiative['title']}** - {initiative['status']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**الوصف:** {initiative['description']}")
                st.write(f"**الأهداف:** {initiative['goals']}")
                st.write(f"**المتطلبات:** {initiative['requirements']}")
                
            with col2:
                st.write(f"**القسم:** {initiative['department']}")
                st.write(f"**الميزانية:** {initiative['budget']} ريال")
                st.write(f"**الحالة:** {initiative['status']}")
                st.write(f"**تاريخ التقديم:** {initiative['created_at']}")
            
            st.markdown("---")
            st.subheader("تحليل الذكاء الاصطناعي")
            st.write(initiative['ai_feedback'])
            
            if initiative['admin_feedback']:
                st.markdown("---")
                st.subheader("ملاحظات الإدارة")
                st.write(initiative['admin_feedback'])

# UI for reviewing initiatives (admin/HR/Finance)
def review_initiatives_page(role):
    st.title(f"👀 مراجعة المبادرات ({role})")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "تصفية حسب الحالة",
            ["الكل", "pending", "approved", "rejected", "in_progress", "implemented"]
        )
    
    with col2:
        department_filter = st.selectbox(
            "تصفية حسب القسم",
            ["الكل", "تقنية المعلومات", "الموارد البشرية", "المالية", "الخدمات", "التطوير", "أخرى"]
        )
    
    with col3:
        budget_filter = st.number_input("الحد الأقصى للميزانية", value=1000000.0, step=10000.0)
    
    # Build query based on filters
    query = "SELECT * FROM initiatives WHERE 1=1"
    params = []
    
    if status_filter != "الكل":
        query += " AND status = ?"
        params.append(status_filter)
    
    if department_filter != "الكل":
        query += " AND department = ?"
        params.append(department_filter)
    
    query += " AND budget <= ?"
    params.append(budget_filter)
    
    query += " ORDER BY created_at DESC"
    
    # Get initiatives based on filters
    initiatives = pd.read_sql_query(query, conn, params=params)
    
    if initiatives.empty:
        st.info("لا توجد مبادرات تطابق معايير التصفية.")
        return
    
    # Display initiatives
    for _, initiative in initiatives.iterrows():
        with st.expander(f"**{initiative['title']}** - {initiative['status']} - {initiative['employee_name']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**الوصف:** {initiative['description']}")
                st.write(f"**الأهداف:** {initiative['goals']}")
                st.write(f"**المتطلبات:** {initiative['requirements']}")
                
            with col2:
                st.write(f"**القسم:** {initiative['department']}")
                st.write(f"**الموظف:** {initiative['employee_name']} ({initiative['employee_id']})")
                st.write(f"**الميزانية:** {initiative['budget']} ريال")
                st.write(f"**تاريخ التقديم:** {initiative['created_at']}")
            
            st.markdown("---")
            st.subheader("تحليل الذكاء الاصطناعي")
            st.write(initiative['ai_feedback'])
            
            # Admin feedback section
            st.markdown("---")
            
            # Add different actions based on role
            if role == "مدير":
                new_status = st.selectbox(
                    "تحديث الحالة",
                    ["pending", "approved", "rejected", "in_progress", "implemented"],
                    key=f"status_{initiative['id']}"
                )
                
                feedback = st.text_area(
                    "إضافة ملاحظات",
                    value=initiative['admin_feedback'] if initiative['admin_feedback'] else "",
                    key=f"feedback_{initiative['id']}"
                )
                
                if st.button("تحديث", key=f"update_{initiative['id']}"):
                    update_initiative_status(initiative['id'], new_status, feedback)
                    st.success("تم تحديث حالة المبادرة")
                    st.experimental_rerun()
            
            elif role == "قسم الموارد البشرية":
                hr_analysis = st.text_area(
                    "تحليل الموارد البشرية",
                    value=initiative['admin_feedback'] if initiative['admin_feedback'] else "",
                    key=f"hr_{initiative['id']}"
                )
                
                if st.button("إضافة تحليل الموارد البشرية", key=f"hr_update_{initiative['id']}"):
                    update_initiative_status(initiative['id'], initiative['status'], hr_analysis)
                    st.success("تم إضافة تحليل الموارد البشرية")
                    st.experimental_rerun()
            
            elif role == "القسم المالي":
                budget_assessment = st.text_area(
                    "تقييم الميزانية",
                    value=initiative['admin_feedback'] if initiative['admin_feedback'] else "",
                    key=f"finance_{initiative['id']}"
                )
                
                adjusted_budget = st.number_input(
                    "الميزانية المعدلة (ريال)",
                    value=float(initiative['budget']),
                    key=f"budget_{initiative['id']}"
                )
                
                if st.button("إضافة تقييم مالي", key=f"finance_update_{initiative['id']}"):
                    feedback = f"تقييم الميزانية: {budget_assessment}\nالميزانية المعدلة: {adjusted_budget} ريال"
                    update_initiative_status(initiative['id'], initiative['status'], feedback)
                    st.success("تم إضافة التقييم المالي")
                    st.experimental_rerun()

# Dashboard for statistics
def dashboard_page():
    st.title("📊 لوحة المعلومات")
    
    # Get data
    initiatives = pd.read_sql_query("SELECT * FROM initiatives", conn)
    
    if initiatives.empty:
        st.info("لا توجد بيانات كافية لعرض الإحصائيات.")
        return
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("إجمالي المبادرات", len(initiatives))
    
    with col2:
        approved = len(initiatives[initiatives['status'] == 'approved'])
        st.metric("المبادرات المعتمدة", approved)
    
    with col3:
        implemented = len(initiatives[initiatives['status'] == 'implemented'])
        st.metric("المبادرات المنفذة", implemented)
    
    with col4:
        total_budget = initiatives['budget'].sum()
        st.metric("إجمالي الميزانيات المقترحة", f"{total_budget:,.0f} ريال")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("المبادرات حسب القسم")
        dept_counts = initiatives['department'].value_counts().reset_index()
        dept_counts.columns = ['القسم', 'العدد']
        st.bar_chart(dept_counts.set_index('القسم'))
    
    with col2:
        st.subheader("المبادرات حسب الحالة")
        status_counts = initiatives['status'].value_counts().reset_index()
        status_counts.columns = ['الحالة', 'العدد']
        st.bar_chart(status_counts.set_index('الحالة'))
    
    # Recent initiatives
    st.subheader("أحدث المبادرات")
    recent = initiatives.sort_values('created_at', ascending=False).head(5)
    
    for _, initiative in recent.iterrows():
        st.write(f"**{initiative['title']}** - {initiative['employee_name']} - {initiative['status']}")

# Add seed data to RAG knowledge base (for demo purposes)
def add_seed_data():
    cursor = conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM rag_knowledge").fetchone()[0]
    
    if count == 0:
        seed_data = [
            {
                "content": "عنوان المبادرة: تطبيق نظام التوقيع الإلكتروني\nالقسم: تقنية المعلومات\nالوصف: إنشاء نظام للتوقيع الإلكتروني للمعاملات الداخلية لتقليل استهلاك الورق وتسريع العمليات\nالأهداف: تقليل البصمة الكربونية، تسريع إنجاز المعاملات، توفير التكاليف\nالمتطلبات: برمجيات تشفير، أجهزة خوادم، تدريب الموظفين\nالميزانية: 150000 ريال",
                "category": "تقنية المعلومات"
            },
            {
                "content": "عنوان المبادرة: برنامج تحفيز الموظفين\nالقسم: الموارد البشرية\nالوصف: برنامج شامل لتحفيز الموظفين من خلال نظام نقاط ومكافآت للإنجازات المتميزة\nالأهداف: زيادة الإنتاجية، تعزيز الانتماء للمؤسسة، تقليل معدل دوران الموظفين\nالمتطلبات: نظام إلكتروني لتتبع النقاط، ميزانية للمكافآت، فريق إدارة البرنامج\nالميزانية: 200000 ريال",
                "category": "الموارد البشرية"
            },
            {
                "content": "عنوان المبادرة: ترشيد استهلاك الطاقة\nالقسم: الخدمات\nالوصف: تركيب أنظمة ذكية لترشيد استهلاك الكهرباء والماء في مباني المؤسسة\nالأهداف: خفض فواتير الطاقة بنسبة 30%، تقليل البصمة البيئية، الالتزام بمعايير الاستدامة\nالمتطلبات: أجهزة استشعار ذكية، أنظمة تحكم مركزية، حملة توعية للموظفين\nالميزانية: 350000 ريال",
                "category": "الخدمات"
            }
        ]
        
        for item in seed_data:
            cursor.execute(
                "INSERT INTO rag_knowledge (content, category) VALUES (?, ?)",
                (item["content"], item["category"])
            )
        
        conn.commit()

# Main function
def main():
    # Add seed data for demo purposes
    add_seed_data()
    
    # Ensure API key is available
    ensure_api_key()
    
    # Handle navigation
    page, role = navigation()
    
    # Display the appropriate page
    if page == "submit_initiative":
        submit_initiative_page()
    
    elif page == "view_my_initiatives":
        view_my_initiatives_page()
    
    elif page == "review_initiatives":
        review_initiatives_page(role)
    
    elif page == "dashboard":
        dashboard_page()

if __name__ == "__main__":
    main()