"""
admin_dashboard.py
Provides a decorated admin dashboard function for the Multi-Disease Prediction app.

Usage (in your main app.py):

from admin_dashboard import show_admin_dashboard

# inside the admin page branch:
show_admin_dashboard(st.session_state.user, _users, MODELS)

Place an optional banner image at: assets/admin_banner.jpg (or png). If the file exists it will be shown.
"""

import streamlit as st
import io
import json
import matplotlib.pyplot as plt
from datetime import datetime


def show_admin_dashboard(current_user, users, models):
    """Render a separated, decorated admin dashboard.

    Args:
        current_user (str): username of the logged-in user (should be 'admin').
        users (dict): full users data structure (from your app's users.json load)
        models (dict): discovered models mapping (key -> {"model": ..., "filename": ...})
    """
    # Security check (call-site should already protect but double-check)
    if not current_user or current_user.lower() != 'admin':
        st.error("Admin dashboard: access denied. Only the admin user may view this page.")
        return

    # Page header with optional image
    banner_path = 'assets/admin_banner.jpg'  # create an assets/ folder and add an image if you want
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("<h1 style='color:#ffffff; margin-bottom:2px;'>Admin Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<div style='color:#bdbdbd; margin-top:-8px;'>Manage users, view prediction history and export data.</div>", unsafe_allow_html=True)
    with cols[1]:
        try:
            st.image(banner_path, width=140)
        except Exception:
            # image optional — ignore failure
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
        fig, ax = plt.subplots(figsize=(8, max(2, len(counts)*0.6)))
        labels = list(counts.keys())
        vals = [counts[k] for k in labels]
        ax.barh(range(len(labels)), vals, color='#ff6b6b')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Count')
        st.pyplot(fig)
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
                    # create CSV-like bytes
                    buf = io.StringIO()
                    buf.write('id,timestamp,model,result,confidence,inputs\n')
                    for rec in history:
                        inputs_json = json.dumps(rec.get('inputs', {})).replace('\n',' ').replace(',', ';')
                        line = f"{rec.get('id')},{rec.get('timestamp')},{rec.get('model')},{rec.get('result')},{rec.get('confidence')},{inputs_json}\n"
                        buf.write(line)
                    st.download_button(f"Download {uname}_history.csv", data=buf.getvalue().encode('utf-8'), file_name=f"{uname}_history.csv", mime='text/csv')

    st.markdown("---")

    # Admin actions: export all users, clear demo data (with confirmation)
    st.markdown("### Admin actions")
    if st.button('Export all users (JSON)'):
        st.download_button('Download users.json', data=json.dumps(users, indent=2), file_name='users_export.json', mime='application/json')

    if st.button('Clear all demo prediction history (keep users)'):
        # destructive: remove history arrays for all users
        for k in users:
            users[k]['history'] = []
        st.success('All histories cleared in memory; remember to save if you want this persisted.')

    st.markdown("---")
    st.markdown("<div style='color:#999;'>Dashboard generated on {}.</div>".format(datetime.now().isoformat()), unsafe_allow_html=True)

    return
