@echo off
rem LiveTranslate portable launcher (plan §4.9: the only surviving launcher).
rem Frozen builds resolve the data root from the exe location automatically
rem (SelfServe P0-A4: <root>\data next to <root>\app), so no env marker is
rem needed here.
start "" "%~dp0app\LiveTranslate.exe"
