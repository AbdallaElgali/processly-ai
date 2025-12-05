
def show_alert(st, status, msg):
    if status == "success":
        st.success("Data successfully processed! 🎉" + '\n' + msg)

    if status == "warning":
        # Displays a yellow box with a warning icon.
        st.warning("Please verify the inputs before proceeding."+ '\n' + msg)

    if status == "error":
        # Displays a red box with an error icon.
        st.error("Operation failed. Check the server logs."+ '\n' + msg)

    if status == "info":
        # Displays a blue box with an info icon.
        st.info("The application is running in demonstration mode."+ '\n' + msg)