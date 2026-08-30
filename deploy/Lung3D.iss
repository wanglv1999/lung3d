; Lung3D 分割重建 安装脚本
#define MyAppName "Lung3D 肺CT三维重建"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "Lung3D"
#define MyAppExe "pythonw.exe"
#define MyAppGUI "lung3d_gui.py"

[Setup]
AppId={{B7E94A2F-3C5E-4A21-9C0D-6F2E4B8A1D5A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Lung3D
DisableProgramGroupPage=yes
DefaultGroupName=Lung3D
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=D:\lung3d_build\installer
OutputBaseFilename=Lung3D-Setup-{#MyAppVersion}
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\.venv\Scripts\pythonw.exe
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName}

[Files]
Source: "D:\lung3d_build\Lung3D\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Lung3D 分割重建"; Filename: "{app}\.venv\Scripts\{#MyAppExe}"; Parameters: "{app}\{#MyAppGUI}"; WorkingDir: "{app}"
Name: "{group}\卸载 Lung3D"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Lung3D 分割重建"; Filename: "{app}\.venv\Scripts\{#MyAppExe}"; Parameters: "{app}\{#MyAppGUI}"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked

[Run]
Filename: "{app}\fix_pyvenv.bat"; WorkingDir: "{app}"; Flags: runhidden; StatusMsg: "正在配置运行环境..."
Filename: "{app}\.venv\Scripts\{#MyAppExe}"; Parameters: "{app}\{#MyAppGUI}"; WorkingDir: "{app}"; Flags: nowait skipifsilent; Description: "立即启动 Lung3D 分割重建"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\.venv\Lib\site-packages\*\__pycache__"