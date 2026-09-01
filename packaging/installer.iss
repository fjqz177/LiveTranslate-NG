; LiveTranslate Windows installer (plan §4.9: Inno Setup 6, per-user).
;
; Build (CI, Windows):  iscc packaging/installer.iss
; The /VERYSILENT switch is the update bridge: a newer setup.exe silently
; reinstalls over the onedir (Inno terminates the running process first,
; avoiding onedir file locks).
;
; SelfServe P0-A4 layout: the onedir bundle lives in {app}\app, ALL runtime
; data (settings/engines/models/transcripts/logs) lives in {app}\data and is
; never touched by installs/updates — data never leaves the install dir.
; Uninstall keeps data\ unless the opt-in task is checked.
;
; tools\uv.exe ships with the app (SelfServe P1-B2): frozen engine installs
; resolve the embedded uv at {app}\tools\uv.exe (core/uv_runner.uv_binary).
; Engines are on-demand — the installer bundles the app + embedded toolchain
; only. The user installs, opens the app, and the runtime engine install
; (core/uv_runner.install_variant) pulls the pinned variant requirements into
; {app}\data\engines on their hardware/network. No preload / offline engine
; component.

#define MyAppName "LiveTranslate"
#ifndef MyAppVersion
  #define MyAppVersion GetEnv("VERSION")
#endif
#define MyAppExeName "LiveTranslate.exe"

[Setup]
AppId={{BB487958-4D0B-4B63-920D-26C4B192DDCC}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=LiveTranslate
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputBaseFilename=LiveTranslate-{#MyAppVersion}-setup
OutputDir=..\dist
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
UninstallDisplayIcon={app}\app\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "deletedata"; Description: "同时删除全部用户数据（设置、模型、转录）"; GroupDescription: "卸载选项"; Flags: unchecked

[Files]
Source: "..\dist\LiveTranslate\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\tools\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
; Bundled CPython (SelfServe P1-B2): the runtime engine installer always uses
; this concrete interpreter — never discovers one on the user machine.
Source: "..\dist\tools\python\*"; DestDir: "{app}\tools\python"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\app\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\app\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\app\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// NOTE: [Code] is PascalScript — ";" is NOT a comment here, only // and {}.
// Keep this file UTF-8 WITH BOM (iscc reads ANSI otherwise).

// Runtime data lives in {app}\data (install-tree data root, SelfServe P0-A4);
// the uninstaller keeps it unless the opt-in task is checked.
//
// WizardIsTaskSelected() cannot be called during uninstall (runtime error
// "Cannot call WizardIsTaskSelected function during Uninstall" — the wizard
// form no longer exists), so the task choice is persisted as a marker file
// at post-install and read back by the uninstaller. The marker is refreshed
// on every (re)install so silent updates honor the latest choice.
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    ForceDirectories(ExpandConstant('{app}\data'));
    if WizardIsTaskSelected('deletedata') then
      SaveStringToFile(ExpandConstant('{app}\data\.delete-data-on-uninstall'), '1', False)
    else
      DeleteFile(ExpandConstant('{app}\data\.delete-data-on-uninstall'));
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then begin
    if FileExists(ExpandConstant('{app}\data\.delete-data-on-uninstall')) then
      DelTree(ExpandConstant('{app}\data'), True, True, True);
  end;
end;
