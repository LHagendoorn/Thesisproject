import smtplib
from email.mime.text import MIMEText

def notify(body):
    msg = MIMEText(body)
    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = 'Finished training!'
    msg['From'] = "scriptnotificiations@gmail.com"
    msg['To'] = "scriptnotificiations@gmail.com"

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("scriptnotificiations@gmail.com", "SuperSecretMegaPassword!!!")
    s.sendmail("scriptnotificiations@gmail.com", ["scriptnotificiations@gmail.com"], msg.as_string())
    s.quit()