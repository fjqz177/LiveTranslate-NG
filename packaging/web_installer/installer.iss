; LiveTranslate web installer stub (SelfServe P2-C5/C6, Windows default channel).
;
; A ~1 MB downloader: probes nothing, writes nothing — it downloads the full
; setup.exe via Windows-native BITS (resumable, no third-party plugins) and
; silently reruns it (/VERYSILENT). When a previous install is detected the
; same flow acts as REPAIR: the full installer re-covers {app}\app and the
; component checksums guard the rest (P2-C6).
;
; Build (CI):  iscc /DMyAppVersion="x.y.z" /DDownloadURL="https://github.com/
;              fjqz177/LiveTranslate/releases/download/vX/LiveTranslate-X-setup.exe"
;              packaging/web_installer/installer.iss
; Output: dist/LiveTranslate-web-setup.exe  (~1 MB)

#define MyAppName "LiveTranslate"
#ifndef MyAppVersion
  #define MyAppVersion GetEnv("VERSION")
#endif
#ifndef DownloadURL
  #define DownloadURL "https://github.com/fjqz177/LiveTranslate/releases/latest"
#endif
#define MainAppId "BB487958-4D0B-4B63-920D-26C4B192DDCC"

[Setup]
AppId={{A7F9E3C1-2B4D-4E6F-8A1C-9D2B5E7F0A3B}
AppName={#MyAppName} Web Installer
AppVersion={#MyAppVersion}
AppPublisher=LiveTranslate
PrivilegesRequired=lowest
CreateAppDir=no
Uninstallable=no
OutputBaseFilename=LiveTranslate-{#MyAppVersion}-web-setup
OutputDir=..\..\dist
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[CustomMessages]
english.Downloading=Downloading the LiveTranslate installer…
english.DownloadFailed=Download failed. Get the full installer manually from:%n{#DownloadURL}
english.RepairNote=An existing LiveTranslate install was detected — this run will repair/update it.
chinesesimplified.Downloading=正在下载 LiveTranslate 安装程序…
chinesesimplified.DownloadFailed=下载失败。请手动下载完整安装包：%n{#DownloadURL}
chinesesimplified.RepairNote=检测到已安装的 LiveTranslate——本次运行将执行修复/更新。

[Code]
var
  RepairMode: Boolean;

function MainAppInstalled(): Boolean;
begin
  Result := RegKeyExists(
    HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#MainAppId}_is1');
end;

procedure InitializeWizard();
begin
  RepairMode := MainAppInstalled();
  if RepairMode then
    MsgBox(ExpandConstant('{cm:RepairNote}'), mbInformation, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  TempFile: string;
  ResultCode: Integer;
begin
  if CurStep <> ssInstall then
    Exit;
  TempFile := ExpandConstant('{tmp}\LiveTranslate-setup.exe');
  WizardForm.StatusLabel.Caption := ExpandConstant('{cm:Downloading}');
  if Exec('powershell.exe',
          '-NoProfile -WindowStyle Hidden -Command "Start-BitsTransfer -Source ''{#DownloadURL}'' -Destination ''' + TempFile + '''"',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and
     FileExists(TempFile) and (ResultCode = 0) then begin
    if Exec(TempFile, '/VERYSILENT /NORESTART', '', SW_SHOWNORMAL,
            ewWaitUntilTerminated, ResultCode) then begin
      WizardForm.Close();
    end;
  end else begin
    MsgBox(ExpandConstant('{cm:DownloadFailed}'), mbError, MB_OK);
    WizardForm.Close();
  end;
end;
