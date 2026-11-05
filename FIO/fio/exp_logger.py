"""Per-run experiment logging.

``start_experiment(exp_name)`` creates ``$EXP_ROOT/<exp_name>/<timestamp>/``
(default ``FIO/experiments/``) containing:
  README.txt   experiment name, EXP_DESC, Slurm job id, start time
  events.log   the root ``logging`` logger is pointed here (INFO level)
and registers an atexit hook that logs the end time / runtime and, if
EXP_NOTIFY_EMAIL and EXP_NOTIFY_APP_PASSWORD are set, sends a best-effort
"job finished" Gmail. Returns the configured logger.
"""

from pathlib import Path
from datetime import datetime
import logging
import os
import atexit

#for mail
import smtplib
from email.message import EmailMessage

# Experiment logs go to $EXP_ROOT/<exp_name>/<timestamp>/ (default FIO/experiments/).
EXP_ROOT = Path(os.environ.get("EXP_ROOT", Path(__file__).resolve().parents[1] / "experiments"))


def send_gmail(subject, body):
    """Best-effort job-finished email. Needs EXP_NOTIFY_EMAIL and
    EXP_NOTIFY_APP_PASSWORD (a Gmail app password) in the environment;
    does nothing when either is unset. Never put credentials in this file."""
    user = os.environ.get("EXP_NOTIFY_EMAIL")
    password = os.environ.get("EXP_NOTIFY_APP_PASSWORD")
    if not user or not password:
        return False

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = user
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
        server.login(user, password)
        server.send_message(msg)
    return True

def start_experiment(exp_name: str):
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    exp_dir = EXP_ROOT / exp_name / timestamp
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
            sent = send_gmail(
                subject=f"SLURM Job Finished: {exp_name}",
                body=(
                    f"Experiment: {exp_name}\n"
                    f"Job ID: {os.getenv('SLURM_JOB_ID')}\n"
                    f"Runtime: {runtime}\n"
                    f"Log: {os.getenv('SLURM_STDOUT')}\n"
                )
            )
            logger.info("Email notification sent" if sent else
                        "Email notification skipped (EXP_NOTIFY_EMAIL / EXP_NOTIFY_APP_PASSWORD unset)")
        except Exception as e:
            logger.info(f"Email notification failed: {e}")

    atexit.register(log_end)

    return logger
