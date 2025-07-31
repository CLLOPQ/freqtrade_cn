Describe "安装与测试" {
  BeforeAll {
    # 安装变量
    $SetupScriptPath = Join-Path $PSScriptRoot "..\setup.ps1"
    $Global:LogFilePath = Join-Path $env:TEMP "script_log.txt"

    # 检查安装脚本是否存在
    if (-Not (Test-Path -Path $SetupScriptPath)) {
      Write-Host "错误: 在路径 $SetupScriptPath 未找到 setup.ps1 脚本"
      exit 1
    }

    # 模拟主函数以阻止其运行
    Mock Main {}

    . $SetupScriptPath
  }

  Context "Write-Log 测试" -Tag "单元" {
    It "应写入 INFO 级别日志" {
      if (Test-Path $Global:LogFilePath){
        Remove-Item $Global:LogFilePath -ErrorAction SilentlyContinue
      }

      Write-Log -Message "测试信息消息" -Level "INFO"
      $Global:LogFilePath | Should -Exist

      $LogContent = Get-Content $Global:LogFilePath
      $LogContent | Should -Contain "INFO: 测试信息消息"
    }

    It "应写入 ERROR 级别日志" {
      if (Test-Path $Global:LogFilePath){
        Remove-Item $Global:LogFilePath -ErrorAction SilentlyContinue
      }

      Write-Log -Message "测试错误消息" -Level "ERROR"
      $Global:LogFilePath | Should -Exist

      $LogContent = Get-Content $Global:LogFilePath
      $LogContent | Should -Contain "ERROR: 测试错误消息"
    }
  }

  Describe "Get-UserSelection 测试" {
    Context "有效输入" {
      It "对于有效的单一选择应返回正确索引" {
        $Options = @("选项1", "选项2", "选项3")
        Mock Read-Host { return "B" }
        $Result = Get-UserSelection -prompt "选择一个选项" -options $Options
        $Result | Should -Be 1
      }

      It "对于有效的小写单一选择应返回正确索引" {
        $Options = @("选项1", "选项2", "选项3")
        Mock Read-Host { return "b" }
        $Result = Get-UserSelection -prompt "选择一个选项" -options $Options
        $Result | Should -Be 1
      }

      It "当未提供输入时应返回默认选择" {
        $Options = @("选项1", "选项2", "选项3")
        Mock Read-Host { return "" }
        $Result = Get-UserSelection -prompt "选择一个选项" -options $Options -defaultChoice "C"
        $Result | Should -Be 2
      }
    }

    Context "无效输入" {
      It "对于无效字母选择应返回 -1" {
        $Options = @("选项1", "选项2", "选项3")
        Mock Read-Host { return "X" }
        $Result = Get-UserSelection -prompt "选择一个选项" -options $Options
        $Result | Should -Be -1
      }

      It "对于超出有效范围的选择应返回 -1" {
        $Options = @("选项1", "选项2", "选项3")
        Mock Read-Host { return "D" }
        $Result = Get-UserSelection -prompt "选择一个选项" -options $Options
        $Result | Should -Be -1
      }

      It "对于非字母输入应返回 -1" {
        $Options = @("选项1", "选项2", "选项3")
        Mock Read-Host { return "1" }
        $Result = Get-UserSelection -prompt "选择一个选项" -options $Options
        $Result | Should -Be -1
      }

      It "对于混合有效和无效输入应返回 -1" {
        Mock Read-Host { return "A,X,B,Y,C,Z" }
        $Options = @("选项1", "选项2", "选项3")
        $Indices = Get-UserSelection -prompt "选择选项" -options $Options -defaultChoice "A"
        $Indices | Should -Be -1
      }
    }

    Context "多项选择" {
      It "应正确处理有效输入" {
        Mock Read-Host { return "A, B, C" }
        $Options = @("选项1", "选项2", "选项3")
        $Indices = Get-UserSelection -prompt "选择选项" -options $Options -defaultChoice "A"
        $Indices | Should -Be @(0, 1, 2)
      }

      It "应正确处理无空格的有效输入" {
        Mock Read-Host { return "A,B,C" }
        $Options = @("选项1", "选项2", "选项3")
        $Indices = Get-UserSelection -prompt "选择选项" -options $Options -defaultChoice "A"
        $Indices | Should -Be @(0, 1, 2)
      }

      It "应返回所选选项的索引" {
        Mock Read-Host { return "a,b" }
        $Options = @("选项1", "选项2", "选项3")
        $Indices = Get-UserSelection -prompt "选择选项" -options $Options
        $Indices | Should -Be @(0, 1)
      }

      It "如果无输入应返回默认选择" {
        Mock Read-Host { return "" }
        $Options = @("选项1", "选项2", "选项3")
        $Indices = Get-UserSelection -prompt "选择选项" -options $Options -defaultChoice "C"
        $Indices | Should -Be @(2)
      }

      It "应优雅处理无效输入" {
        Mock Read-Host { return "x,y,z" }
        $Options = @("选项1", "选项2", "选项3")
        $Indices = Get-UserSelection -prompt "选择选项" -options $Options -defaultChoice "A"
        $Indices | Should -Be -1
      }

      It "应处理无空格输入" {
        Mock Read-Host { return "a,b,c" }
        $Options = @("选项1", "选项2", "选项3")
        $Indices = Get-UserSelection -prompt "选择选项" -options $Options
        $Indices | Should -Be @(0, 1, 2)
      }
    }
  }

  Describe "Exit-Script 测试" -Tag "单元" {
    BeforeEach {
      Mock Write-Log {}
      Mock Start-Process {}
      Mock Read-Host { return "Y" }
    }

    It "应使用给定的退出代码退出，无需等待按键" {
      $ExitCode = Exit-Script -ExitCode 0 -isSubShell $true -waitForKeypress $false
      $ExitCode | Should -Be 0
    }

    It "在错误时应提示打开日志文件" {
      Exit-Script -ExitCode 1 -isSubShell $true -waitForKeypress $false
      Assert-MockCalled Read-Host -Exactly 1
      Assert-MockCalled Start-Process -Exactly 1
    }
  }

  Context 'Find-PythonExecutable' {
    It '返回第一个有效的Python可执行文件' {
      Mock Test-PythonExecutable { $true } -ParameterFilter { $PythonExecutable -eq 'python' }
      $Result = Find-PythonExecutable
      $Result | Should -Be 'python'
    }

    It '如果未找到有效的Python可执行文件则返回null' {
      Mock Test-PythonExecutable { $false }
      $Result = Find-PythonExecutable
      $Result | Should -Be $null
    }
  }
}