Clear-Host

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Global:LogFilePath = Join-Path $env:TEMP "script_log_$Timestamp.txt"

$RequirementFiles = @("requirements.txt", "requirements-dev.txt", "requirements-hyperopt.txt", "requirements-freqai.txt", "requirements-freqai-rl.txt", "requirements-plot.txt")
$VenvName = ".venv"
$VenvDir = Join-Path $PSScriptRoot $VenvName

function Write-Log {
  param (
    [string]$Message,
    [string]$Level = 'INFO'
  )

  if (-not (Test-Path -Path $LogFilePath)) {
    New-Item -ItemType File -Path $LogFilePath -Force | Out-Null
  }

  switch ($Level) {
    'INFO' { Write-Host $Message -ForegroundColor Green }
    'WARNING' { Write-Host $Message -ForegroundColor Yellow }
    'ERROR' { Write-Host $Message -ForegroundColor Red }
    'PROMPT' { Write-Host $Message -ForegroundColor Cyan }
  }

  "${Level}: $Message" | Out-File $LogFilePath -Append
}

function Get-UserSelection {
  param (
    [string]$Prompt,
    [string[]]$Options,
    [string]$DefaultChoice = 'A',
    [bool]$AllowMultipleSelections = $true
  )

  Write-Log "$Prompt`n" -Level 'PROMPT'
  for ($I = 0; $I -lt $Options.Length; $I++) {
    Write-Log "$([char](65 + $I)). $($Options[$I])" -Level 'PROMPT'
  }

  if ($AllowMultipleSelections) {
    Write-Log "`n请通过输入相应字母（用逗号分隔）选择一个或多个选项。" -Level 'PROMPT'
  }
  else {
    Write-Log "`n请通过输入相应字母选择一个选项。" -Level 'PROMPT'
  }

  [string]$UserInput = Read-Host
  if ([string]::IsNullOrEmpty($UserInput)) {
    $UserInput = $DefaultChoice
  }
  $UserInput = $UserInput.ToUpper()

  if ($AllowMultipleSelections) {
    $Selections = $UserInput.Split(',') | ForEach-Object { $_.Trim() }
    $SelectedIndices = @()
    foreach ($Selection in $Selections) {
      if ($Selection -match '^[A-Z]$') {
        $Index = [int][char]$Selection - [int][char]'A'
        if ($Index -ge 0 -and $Index -lt $Options.Length) {
          $SelectedIndices += $Index
        }
        else {
          Write-Log "无效输入：$Selection。请输入在有效选项范围内的字母。" -Level 'ERROR'
          return -1
        }
      }
      else {
        Write-Log "无效输入：$Selection。请输入 A 到 Z 之间的字母。" -Level 'ERROR'
        return -1
      }
    }
    return $SelectedIndices
  }
  else {
    if ($UserInput -match '^[A-Z]$') {
      $SelectedIndex = [int][char]$UserInput - [int][char]'A'
      if ($SelectedIndex -ge 0 -and $SelectedIndex -lt $Options.Length) {
        return $SelectedIndex
      }
      else {
        Write-Log "无效输入：$UserInput。请输入在有效选项范围内的字母。" -Level 'ERROR'
        return -1
      }
    }
    else {
      Write-Log "无效输入：$UserInput。请输入 A 到 Z 之间的字母。" -Level 'ERROR'
      return -1
    }
  }
}

function Exit-Script {
  param (
    [int]$ExitCode,
    [bool]$WaitForKeypress = $true
  )

  if ($ExitCode -ne 0) {
    Write-Log "脚本执行失败。是否要打开日志文件？(Y/N)" -Level 'PROMPT'
    $openLog = Read-Host
    if ($openLog -eq 'Y' -or $openLog -eq 'y') {
      Start-Process notepad.exe -ArgumentList $LogFilePath
    }
  }
  elseif ($WaitForKeypress) {
    Write-Log "按任意键退出..."
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
  }

  return $ExitCode
}

function Test-PythonExecutable {
  param(
    [string]$PythonExecutable
  )

  $DeactivateVenv = Join-Path $VenvDir "Scripts\Deactivate.bat"
  if (Test-Path $DeactivateVenv) {
    Write-Host "正在退出虚拟环境..." 2>&1 | Out-File $LogFilePath -Append
    & $DeactivateVenv
    Write-Host "虚拟环境已退出。" 2>&1 | Out-File $LogFilePath -Append
  }
  else {
    Write-Host "未找到退出脚本：$DeactivateVenv" 2>&1 | Out-File $LogFilePath -Append
  }

  $PythonCmd = Get-Command $PythonExecutable -ErrorAction SilentlyContinue
  if ($PythonCmd) {
    $VersionOutput = & $PythonCmd.Source --version 2>&1
    if ($LASTEXITCODE -eq 0) {
      $Version = $VersionOutput | Select-String -Pattern "Python (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
      Write-Log "使用可执行文件 '$PythonExecutable' 找到了 Python 版本 $Version。"
      return $true
    }
    else {
      Write-Log "Python 可执行文件 '$PythonExecutable' 无法正常工作。" -Level 'ERROR'
      return $false
    }
  }
  else {
    Write-Log "未找到 Python 可执行文件 '$PythonExecutable'。" -Level 'ERROR'
    return $false
  }
}

function Find-PythonExecutable {
  $PythonExecutables = @(
    "python",
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.10",
    "python3",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python310\python.exe",
    "C:\Python313\python.exe",
    "C:\Python312\python.exe",
    "C:\Python311\python.exe",
    "C:\Python310\python.exe"
  )


  foreach ($Executable in $PythonExecutables) {
    if (Test-PythonExecutable -PythonExecutable $Executable) {
      return $Executable
    }
  }

  return $null
}
function Main {
  "开始操作..." | Out-File $LogFilePath -Append
  "当前目录：$(Get-Location)" | Out-File $LogFilePath -Append

  # 当 Python 版本低于 3.10 或未找到 Python 可执行文件时退出
  $PythonExecutable = Find-PythonExecutable
  if ($null -eq $PythonExecutable) {
    Write-Log "未找到合适的 Python 可执行文件。请确保已安装 Python 3.10 或更高版本，并且在系统 PATH 中可用。" -Level 'ERROR'
    Exit 1
  }

  # 定义虚拟环境中 Python 可执行文件的路径
  $ActivateVenv = "$VenvDir\Scripts\Activate.ps1"

  # 检查虚拟环境是否存在，如果不存在则创建
  if (-Not (Test-Path $ActivateVenv)) {
    Write-Log "未找到虚拟环境。正在创建虚拟环境..." -Level 'ERROR'
    & $PythonExecutable -m venv $VenvName 2>&1 | Out-File $LogFilePath -Append
    if ($LASTEXITCODE -ne 0) {
      Write-Log "创建虚拟环境失败。" -Level 'ERROR'
      Exit-Script -exitCode 1
    }
    else {
      Write-Log "虚拟环境已创建。"
    }
  }

  # 激活虚拟环境并检查是否成功
  Write-Log "找到虚拟环境。正在激活虚拟环境..."
  & $ActivateVenv 2>&1 | Out-File $LogFilePath -Append
  # 检查虚拟环境是否已激活
  if ($env:VIRTUAL_ENV) {
    Write-Log "虚拟环境已在以下位置激活：$($env:VIRTUAL_ENV)"
  }
  else {
    Write-Log "激活虚拟环境失败。" -Level 'ERROR'
    Exit-Script -exitCode 1
  }

  # 确保 pip 已安装
  python -m ensurepip --default-pip 2>&1 | Out-File $LogFilePath -Append

  # 仅当仓库状态干净时拉取最新更新
  Write-Log "检查仓库是否干净..."
  $Status = & "git" status --porcelain
  if ($Status) {
    Write-Log "本地 git 仓库中有更改。跳过 git pull。"
  }
  else {
    Write-Log "拉取最新更新..."
    & "git" pull 2>&1 | Out-File $LogFilePath -Append
    if ($LASTEXITCODE -ne 0) {
      Write-Log "从 Git 拉取更新失败。" -Level 'ERROR'
      Exit-Script -exitCode 1
    }
  }

  if (-not (Test-Path "$VenvDir\Lib\site-packages\talib")) {
    # 使用虚拟环境的 pip 安装 TA-Lib
    Write-Log "使用虚拟环境的 pip 安装 TA-Lib..."
    python -m pip install --find-links=build_helpers\ --prefer-binary TA-Lib 2>&1 | Out-File $LogFilePath -Append
    if ($LASTEXITCODE -ne 0) {
      Write-Log "安装 TA-Lib 失败。" -Level 'ERROR'
      Exit-Script -exitCode 1
    }
  }

  # 显示需求文件选项
  $SelectedIndices = Get-UserSelection -prompt "选择要安装的需求文件：" -options $RequirementFiles -defaultChoice 'A'

  # 缓存选中的需求文件
  $SelectedRequirementFiles = @()
  $PipInstallArguments = @()
  foreach ($Index in $SelectedIndices) {
    $RelativePath = $RequirementFiles[$Index]
    if (Test-Path $RelativePath) {
      $SelectedRequirementFiles += $RelativePath
      $PipInstallArguments += "-r", $RelativePath  # 将每个标志和路径作为单独的元素添加
    }
    else {
      Write-Log "未找到需求文件：$RelativePath" -Level 'ERROR'
      Exit-Script -exitCode 1
    }
  }
  if ($PipInstallArguments.Count -ne 0) {
    & pip install @PipInstallArguments  # 使用数组展开正确传递参数
  }

  # 使用虚拟环境的 Python 从 setup 安装 freqtrade
  Write-Log "从 setup 安装 freqtrade..."
  pip install -e . 2>&1 | Out-File $LogFilePath -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Log "安装 freqtrade 失败。" -Level 'ERROR'
    Exit-Script -exitCode 1
  }

  Write-Log "安装 freqUI..."
  python freqtrade install-ui 2>&1 | Out-File $LogFilePath -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Log "安装 freqUI 失败。" -Level 'ERROR'
    Exit-Script -exitCode 1
  }

  Write-Log "安装/更新完成！"
  Exit-Script -exitCode 0
}

# 调用 Main 函数
Main