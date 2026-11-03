from pathlib import Path
from datetime import datetime
import logging
import os
import atexit

#for mail
import smtplib
from email.message import EmailMessage

PROJECT_ROOT = Path("/home/rs/24CS92R01/EGOLIFE/EgoLife_copy")

#--for mail notification test
def send_gmail(subject, body):
    GMAIL_USER = "iskousik@gmail.com"
    GMAIL_APP_PASSWORD = "psam fhyc jjea fabi"  # app password

    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_USER
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)

def start_experiment(exp_name: str):
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    exp_dir = PROJECT_ROOT / "ZZexperiments" / exp_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_desc = os.getenv("EXP_DESC", "No description provided")
    slurm_job_id = os.getenv("SLURM_JOB_ID")

    # write README
    with open(exp_dir / "README.txt", "w") as f:
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Description: {exp_desc}\n")
        f.write(f"SLURM Job ID: {slurm_job_id}\n")
        f.write(f"Start Time: {start_time}\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(exp_dir / "events.log")]
    )

    logger = logging.getLogger()
    logger.info(f"Experiment dir: {exp_dir}")
    logger.info(f"Experiment description: {exp_desc}")
    logger.info(f"SLURM_JOB_ID: {slurm_job_id}")
    logger.info(f"SLURM_STDOUT: {os.getenv('SLURM_STDOUT')}")
    logger.info(f"SLURM_STDERR: {os.getenv('SLURM_STDERR')}")

    def log_end():
        end_time = datetime.now()
        runtime = end_time - start_time
        logger.info(f"End time: {end_time}")
        logger.info(f"Total runtime: {end_time - start_time}")
       # Email (best effort)
        try:
            send_gmail(
                subject=f"SLURM Job Finished: {exp_name}",
                body=(
                    f"Experiment: {exp_name}\n"
                    f"Job ID: {os.getenv('SLURM_JOB_ID')}\n"
                    f"Runtime: {runtime}\n"
                    f"Log: {os.getenv('SLURM_STDOUT')}\n"
                )
            )
            logger.info("Email notification sent")
        except Exception as e:
            logger.info(f"Email notification failed: {e}")

    atexit.register(log_end)

    return logger
