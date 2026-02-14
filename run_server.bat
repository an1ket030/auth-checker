@echo off
echo STARTING AUTH-CHECKER SERVER on 0.0.0.0:8000
echo Your IP is: 11.12.4.147
echo ---------------------------------------------------
echo Keep this window OPEN while testing the app on your phone.
echo ---------------------------------------------------

REM Do NOT cd into backend. We run from the root so "backend" is treated as a package.
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
pause
