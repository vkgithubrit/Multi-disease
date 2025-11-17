# admin_dashboard.py
"""
Admin dashboard without matplotlib dependency.
Usage:
    from admin_dashboard import show_admin_dashboard
    show_admin_dashboard(current_user, users, models)
"""

import streamlit as st
import io
import json
from datetime import datetime

def show_admin_dashboard(current_user, users, models):
    """Render a decorated admin dashboard without matplotlib."""
    # Security check
    if not current_user or current_user.lower() != 'admin':
        st.error("Admin dashboard: access denied. Only the admin user may view this page.")
        return

    # Header + optional image
    banner_path = 'assets/admin_banner.jpg'
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("<h1 style='color:#ffffff; margin-bottom:2px;'>Admin Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<div style='color:#bdbdbd; margin-top:-8px;'>Manage users, view prediction history and export data.</div>", unsafe_allow_html=True)
    with cols[1]:
        try:
            st.image(banner_path, width=140)
        except Exception:
            st.write("")

    st.markdown("---")

    # Summary metrics
    total_users = len(users)
    total_predictions = sum(len(u.get('history', [])) for u in users.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total users", total_users)
    c2.metric("Total predictions", total_predictions)
    c3.metric("Loaded models", len(models))

    # Predictions per model
    counts = {}
    for u in users.values():
        for rec in u.get('history', []):
            m = rec.get('model', 'unknown')
            counts[m] = counts.get(m, 0) + 1

    st.markdown("### Predictions by model")
    if counts:
        # Use Streamlit's native chart or a simple table
        model_names = list(counts.keys())
        model_values = [counts[k] for k in model_names]
        rows = [{"model": m, "count": counts[m]} for m in model_names]
        st.table(rows)
        try:
            import pandas as pd
            df = pd.DataFrame({"model": model_names, "count": model_values}).set_index("model")
            st.bar_chart(df)
        except Exception:
            for m, v in counts.items():
                st.write(f"- {m}: {v}")
    else:
        st.info('No prediction records yet.')

    st.markdown("---")

    # Users table and per-user expanders
    st.markdown("### Registered Users")
    for uname, u in users.items():
        with st.expander(f"{uname} — {u.get('name','')} — {len(u.get('history',[]))} predictions"):
            st.write("**Profile**")
            st.write(f"Name: {u.get('name','')}\n\nEmail: {u.get('email','')}")
            st.write("**Recent predictions (newest first)**")
            history = u.get('history', [])
            if not history:
                st.info('No predictions')
            else:
                for rec in history[:20]:
                    st.write(f"- {rec.get('timestamp')} — {rec.get('model')} — {rec.get('result')} (conf: {rec.get('confidence')})")
                if st.button(f"Export full history for {uname}", key=f"export_{uname}"):
                    buf = io.StringIO()
                    buf.write('id,timestamp,model,result,confidence,inputs\n')
                    for rec in history:
                        inputs_json = json.dumps(rec.get('inputs', {})).replace('\n',' ').replace(',', ';')
                        line = f"{rec.get('id')},{rec.get('timestamp')},{rec.get('model')},{rec.get('result')},{rec.get('confidence')},{inputs_json}\n"
                        buf.write(line)
                    st.download_button(f"Download {uname}_history.csv", data=buf.getvalue().encode('utf-8'), file_name=f"{uname}_history.csv", mime='text/csv')

    st.markdown("---")

    # Admin actions
    st.markdown("### Admin actions")
    if st.button('Export all users (JSON)'):
        st.download_button('Download users.json', data=json.dumps(users, indent=2), file_name='users_export.json', mime='application/json')

    if st.button('Clear all demo prediction history (keep users)'):
        for k in users:
            users[k]['history'] = []
        st.success('All histories cleared in memory; remember to save if you want this persisted.')

    st.markdown("---")
    st.markdown(f"<div style='color:#999;'>Dashboard generated on {datetime.now().isoformat()}.</div>", unsafe_allow_html=True)
    return
