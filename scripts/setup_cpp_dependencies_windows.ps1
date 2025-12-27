# =============================================================================
# SCL Core - Windows Dependencies Setup Script (PowerShell)
# =============================================================================
#
# This script installs system-level dependencies for building scl-core on Windows.
# Requires: Windows 10/11, PowerShell 5.1+, Administrator privileges
#
# Usage:
#   .\scripts\setup_cpp_dependencies_windows.ps1 [-Backend <backend>]
#
# Backend options:
#   - openmp  : Install OpenMP (default, included with MSVC/MinGW)
#   - tbb     : Install Intel TBB via vcpkg
#   - bs      : No system dependencies (header-only)
#   - serial  : No system dependencies
#
# Dependencies installed:
#   - Visual Studio Build Tools 2022 or Visual Studio 2022
#   - CMake 3.15+
#   - Ninja build system
#   - vcpkg (optional, for TBB and HDF5)
#   - Git
#
# =============================================================================

param(
    [Parameter(Mandatory=$false)]
    [string]$Backend = "openmp"
)

# Require Administrator
#Requires -RunAsAdministrator

# =============================================================================
# Helper Functions
# =============================================================================

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Command {
    param([string]$Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    } catch {
        return $false
    }
}

function Test-Chocolatey {
    if (Test-Command choco) {
        Write-Success "Chocolatey installed: $(choco --version)"
        return $true
    } else {
        Write-Warning "Chocolatey not found"
        return $false
    }
}

function Install-Chocolatey {
    Write-Info "Installing Chocolatey package manager..."
    
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    
    try {
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Success "Chocolatey installed successfully"
        
        # Refresh environment
        refreshenv
        return $true
    } catch {
        Write-Error-Custom "Failed to install Chocolatey: $_"
        return $false
    }
}

# =============================================================================
# Visual Studio / Build Tools Detection
# =============================================================================

function Test-VisualStudio {
    Write-Info "Checking for Visual Studio or Build Tools..."
    
    # Check for vswhere (installed with VS 2017+)
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -property installationPath
        
        if ($vsPath) {
            $vsVersion = & $vswhere -latest -property catalog_productDisplayVersion
            Write-Success "Visual Studio or Build Tools found: Version $vsVersion"
            Write-Info "Installation path: $vsPath"
            return $true
        }
    }
    
    Write-Warning "Visual Studio or Build Tools not found"
    return $false
}

function Install-VisualStudioBuildTools {
    Write-Info "Visual Studio Build Tools are required for C++ compilation on Windows"
    Write-Info ""
    Write-Info "Options:"
    Write-Info "  1. Install Visual Studio 2022 Community (recommended, includes IDE)"
    Write-Info "  2. Install Visual Studio Build Tools 2022 (command-line only, smaller)"
    Write-Info "  3. Skip (I have Visual Studio installed already)"
    Write-Info ""
    
    $choice = Read-Host "Choose (1/2/3)"
    
    switch ($choice) {
        "1" {
            Write-Info "Installing Visual Studio 2022 Community..."
            choco install visualstudio2022community --params "--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --passive" -y
            Write-Success "Visual Studio 2022 Community installed"
        }
        "2" {
            Write-Info "Installing Visual Studio Build Tools 2022..."
            choco install visualstudio2022buildtools --params "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive" -y
            Write-Success "Build Tools installed"
        }
        "3" {
            Write-Info "Skipping Visual Studio installation"
        }
        default {
            Write-Error-Custom "Invalid choice"
            exit 1
        }
    }
}

# =============================================================================
# Dependency Installation
# =============================================================================

function Install-BuildTools {
    Write-Info "Installing build tools..."
    
    # Install CMake
    if (-not (Test-Command cmake)) {
        choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' -y
        Write-Success "CMake installed"
    } else {
        $cmakeVersion = cmake --version | Select-String -Pattern "\d+\.\d+\.\d+" | ForEach-Object { $_.Matches.Value }
        Write-Info "CMake $cmakeVersion already installed"
    }
    
    # Install Ninja
    if (-not (Test-Command ninja)) {
        choco install ninja -y
        Write-Success "Ninja installed"
    } else {
        Write-Info "Ninja already installed"
    }
    
    # Install Git
    if (-not (Test-Command git)) {
        choco install git -y
        Write-Success "Git installed"
    } else {
        Write-Info "Git already installed"
    }
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

function Install-Vcpkg {
    Write-Info "Installing vcpkg package manager..."
    
    $vcpkgPath = "C:\vcpkg"
    
    if (Test-Path $vcpkgPath) {
        Write-Info "vcpkg already installed at $vcpkgPath"
        
        # Update vcpkg
        Push-Location $vcpkgPath
        git pull
        .\bootstrap-vcpkg.bat
        Pop-Location
    } else {
        # Clone vcpkg
        git clone https://github.com/Microsoft/vcpkg.git $vcpkgPath
        
        # Bootstrap vcpkg
        Push-Location $vcpkgPath
        .\bootstrap-vcpkg.bat
        Pop-Location
        
        # Add to PATH
        [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$vcpkgPath", [System.EnvironmentVariableTarget]::Machine)
        $env:Path += ";$vcpkgPath"
    }
    
    Write-Success "vcpkg installed at $vcpkgPath"
    Write-Info "Set VCPKG_ROOT environment variable"
    [System.Environment]::SetEnvironmentVariable("VCPKG_ROOT", $vcpkgPath, [System.EnvironmentVariableTarget]::Machine)
    $env:VCPKG_ROOT = $vcpkgPath
}

function Install-TBB {
    Write-Info "Installing Intel TBB via vcpkg..."
    
    if (-not $env:VCPKG_ROOT) {
        Install-Vcpkg
    }
    
    & "$env:VCPKG_ROOT\vcpkg.exe" install tbb:x64-windows
    
    Write-Success "Intel TBB installed"
    Write-Info "CMake will find TBB via vcpkg toolchain file"
}

function Install-HDF5 {
    Write-Info "Installing HDF5 via vcpkg..."
    
    if (-not $env:VCPKG_ROOT) {
        Install-Vcpkg
    }
    
    & "$env:VCPKG_ROOT\vcpkg.exe" install hdf5:x64-windows
    
    Write-Success "HDF5 installed"
}

# =============================================================================
# Main
# =============================================================================

function Main {
    Write-Host "=========================================="
    Write-Host "SCL Core - Windows Dependencies Setup"
    Write-Host "=========================================="
    Write-Host ""
    
    # Check/Install Chocolatey
    if (-not (Test-Chocolatey)) {
        $install = Read-Host "Install Chocolatey package manager? (Y/n)"
        if ($install -ne "n" -and $install -ne "N") {
            if (-not (Install-Chocolatey)) {
                Write-Error-Custom "Failed to install Chocolatey"
                exit 1
            }
        } else {
            Write-Error-Custom "Chocolatey is required for dependency management"
            exit 1
        }
    }
    
    # Check/Install Visual Studio
    if (-not (Test-VisualStudio)) {
        Install-VisualStudioBuildTools
    }
    
    # Install build tools
    Install-BuildTools
    
    # Install HDF5
    $installHdf5 = Read-Host "Install HDF5 for .h5ad support? (Y/n)"
    if ($installHdf5 -ne "n" -and $installHdf5 -ne "N") {
        Install-HDF5
    } else {
        Write-Warning "Skipping HDF5"
    }
    
    # Install threading backend
    switch ($Backend.ToLower()) {
        "openmp" {
            Write-Info "Using OpenMP backend (included with MSVC)"
            Write-Success "OpenMP is included with Visual Studio - no additional installation needed"
        }
        "tbb" {
            Install-TBB
        }
        "bs" {
            Write-Info "Using BS::thread_pool backend (header-only, no dependencies)"
            Write-Success "No system dependencies required for BS backend"
        }
        "serial" {
            Write-Info "Using serial backend (no threading dependencies)"
        }
        default {
            Write-Error-Custom "Unknown backend: $Backend"
            Write-Host "Supported: openmp, tbb, bs, serial"
            exit 1
        }
    }
    
    Write-Host ""
    Write-Host "=========================================="
    Write-Success "Dependencies installed!"
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Open x64 Native Tools Command Prompt for VS 2022"
    Write-Host "  2. Navigate to project directory"
    Write-Host "  3. Run:"
    Write-Host ""
    
    if ($Backend.ToLower() -eq "tbb" -or (Read-Host "Installed HDF5? (y/N)") -eq "y") {
        Write-Host "     cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=`"C:\vcpkg\scripts\buildsystems\vcpkg.cmake`""
    } else {
        Write-Host "     cmake -B build -G Ninja"
    }
    
    if ($Backend.ToLower() -eq "tbb") {
        Write-Host "     -DSCL_THREADING_BACKEND=TBB"
    }
    
    Write-Host ""
    Write-Host "     cmake --build build --config Release"
    Write-Host ""
    
    Write-Info "Note: OpenMP is enabled by default on Windows with MSVC"
    Write-Info "      No additional configuration needed for OpenMP backend"
    Write-Host ""
}

# Run main function
Main

