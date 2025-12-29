#pragma once

// =============================================================================
// SCL Core - Advanced Test Registration and Execution Framework
// =============================================================================
//
// A modern, feature-rich C++ test framework with pytest-style output.
//
// Features:
//   ✓ Auto-registration via __COUNTER__
//   ✓ Rich CLI with extensive options
//   ✓ Modern pytest-style colored output
//   ✓ Multiple output formats (TAP, JSON, JUnit XML, HTML, Markdown, GitHub Actions)
//   ✓ Test fixtures (setup/teardown)
//   ✓ Test suites and grouping
//   ✓ Parameterized tests
//   ✓ Skip, expected failure markers
//   ✓ Rich assertion macros with diff output
//   ✓ Timeout support
//   ✓ Test retries
//   ✓ Random order execution
//   ✓ Parallel execution hints
//   ✓ Progress bar
//   ✓ Performance benchmarking
//   ✓ Signal handling for crashes
//   ✓ Memory tracking
//   ✓ Tags/labels for filtering
//
// Usage:
//   SCL_TEST_BEGIN
//
//   SCL_TEST_UNIT(my_test) {
//       SCL_ASSERT_EQ(1 + 1, 2);
//   }
//
//   SCL_TEST_SUITE(math_tests) {
//       SCL_TEST_CASE(addition) { ... }
//       SCL_TEST_CASE(subtraction) { ... }
//   }
//
//   SCL_TEST_END
//   SCL_TEST_MAIN()
//
// CLI:
//   ./test --help                     # Show all options
//   ./test --filter "transpose"       # Filter by name pattern
//   ./test --tag unit                 # Run tests with specific tag
//   ./test --tap                      # TAP format output
//   ./test --json report.json         # JSON report
//   ./test --xml results.xml          # JUnit XML report
//   ./test --html report.html         # HTML report
//   ./test --github                   # GitHub Actions format
//   ./test --fail-fast                # Stop on first failure
//   ./test --shuffle                  # Random test order
//   ./test --repeat 3                 # Repeat each test
//   ./test --timeout 5000             # Default timeout (ms)
//   ./test -v                         # Verbose output
//   ./test -q                         # Quiet mode
//
// =============================================================================

#ifndef SCL_TEST_HPP
#define SCL_TEST_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#define SCL_UNIX_LIKE 1
#else
#define SCL_UNIX_LIKE 0
#endif

// =============================================================================
// Configuration
// =============================================================================

namespace scl::test {

constexpr std::size_t MAX_TEST_UNITS = 1024;
constexpr std::size_t MAX_SUITES = 64;
constexpr int DEFAULT_TIMEOUT_MS = 30000;
constexpr int DEFAULT_TERMINAL_WIDTH = 80;

} // namespace scl::test

// =============================================================================
// Forward Declarations
// =============================================================================

namespace scl::test {
    class TestException;
    class SkipException;
    class TimeoutException;
    struct TestInfo;
    struct SuiteInfo;
    struct TestResult;
    struct Config;
    class Runner;
    class Reporter;
}

// =============================================================================
// ANSI Color Codes (Modern Palette)
// =============================================================================

namespace scl::test::color {

// Reset
inline const char* reset()      { return "\033[0m"; }

// Modern color palette
inline const char* bold()       { return "\033[1m"; }
inline const char* dim()        { return "\033[2m"; }
inline const char* italic()     { return "\033[3m"; }
inline const char* underline()  { return "\033[4m"; }

// Foreground colors - Modern palette
inline const char* black()      { return "\033[30m"; }
inline const char* red()        { return "\033[38;5;203m"; }      // Soft red
inline const char* green()      { return "\033[38;5;114m"; }      // Soft green  
inline const char* yellow()     { return "\033[38;5;221m"; }      // Warm yellow
inline const char* blue()       { return "\033[38;5;75m"; }       // Sky blue
inline const char* magenta()    { return "\033[38;5;176m"; }      // Soft magenta
inline const char* cyan()       { return "\033[38;5;80m"; }       // Soft cyan
inline const char* white()      { return "\033[97m"; }
inline const char* gray()       { return "\033[38;5;245m"; }      // Medium gray
inline const char* light_gray() { return "\033[38;5;250m"; }

// Bright variants
inline const char* bright_red()    { return "\033[38;5;196m"; }
inline const char* bright_green()  { return "\033[38;5;46m"; }
inline const char* bright_yellow() { return "\033[38;5;226m"; }
inline const char* bright_blue()   { return "\033[38;5;39m"; }

// Background colors
inline const char* bg_red()     { return "\033[48;5;52m"; }
inline const char* bg_green()   { return "\033[48;5;22m"; }
inline const char* bg_yellow()  { return "\033[48;5;58m"; }
inline const char* bg_blue()    { return "\033[48;5;17m"; }
inline const char* bg_gray()    { return "\033[48;5;236m"; }

// Status colors (pytest-style)
inline const char* passed()     { return "\033[38;5;114m"; }      // Green
inline const char* failed()     { return "\033[38;5;203m"; }      // Red  
inline const char* skipped()    { return "\033[38;5;221m"; }      // Yellow
inline const char* xfail()      { return "\033[38;5;221m"; }      // Yellow (expected fail)
inline const char* xpass()      { return "\033[38;5;203m"; }      // Red (unexpected pass)
inline const char* error()      { return "\033[38;5;196m"; }      // Bright red
inline const char* warning()    { return "\033[38;5;214m"; }      // Orange

// Progress bar colors
inline const char* progress_done()    { return "\033[38;5;114m"; }
inline const char* progress_fail()    { return "\033[38;5;203m"; }
inline const char* progress_pending() { return "\033[38;5;240m"; }

} // namespace scl::test::color

// =============================================================================
// Utility Functions
// =============================================================================

namespace scl::test::util {

inline int get_terminal_width() {
#if SCL_UNIX_LIKE
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
        return w.ws_col;
    }
#endif
    return DEFAULT_TERMINAL_WIDTH;
}

inline bool is_tty() {
#if SCL_UNIX_LIKE
    return isatty(STDOUT_FILENO) != 0;
#else
    return false;
#endif
}

inline std::string escape_xml(const std::string& s) {
    std::string result;
    result.reserve(s.size() * 1.1);
    for (char c : s) {
        switch (c) {
            case '&':  result += "&amp;"; break;
            case '<':  result += "&lt;"; break;
            case '>':  result += "&gt;"; break;
            case '"':  result += "&quot;"; break;
            case '\'': result += "&apos;"; break;
            default:   result += c; break;
        }
    }
    return result;
}

inline std::string escape_json(const std::string& s) {
    std::string result;
    result.reserve(s.size() * 1.1);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:   result += c; break;
        }
    }
    return result;
}

inline std::string escape_html(const std::string& s) {
    return escape_xml(s);
}

inline std::string format_duration(double ms) {
    if (ms < 1.0) {
        return std::to_string(static_cast<int>(ms * 1000)) + "µs";
    } else if (ms < 1000.0) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2fms", ms);
        return buf;
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2fs", ms / 1000.0);
        return buf;
    }
}

inline std::string format_bytes(std::size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 3) {
        size /= 1024.0;
        unit++;
    }
    char buf[32];
    if (unit == 0) {
        std::snprintf(buf, sizeof(buf), "%zu %s", bytes, units[unit]);
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit]);
    }
    return buf;
}

inline std::string timestamp_iso8601() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::gmtime(&time));
    
    char result[80];
    std::snprintf(result, sizeof(result), "%s.%03ldZ", buf, static_cast<long>(ms.count()));
    return result;
}

inline std::string repeat_string(const std::string& s, int count) {
    std::string result;
    result.reserve(s.size() * count);
    for (int i = 0; i < count; ++i) {
        result += s;
    }
    return result;
}

inline std::string truncate(const std::string& s, std::size_t max_len, const std::string& suffix = "...") {
    if (s.length() <= max_len) return s;
    if (max_len <= suffix.length()) return suffix.substr(0, max_len);
    return s.substr(0, max_len - suffix.length()) + suffix;
}

// String diff for assertion failures
inline std::pair<std::string, std::string> compute_diff(const std::string& expected, const std::string& actual) {
    // Find first difference
    std::size_t diff_start = 0;
    while (diff_start < expected.size() && diff_start < actual.size() && 
           expected[diff_start] == actual[diff_start]) {
        diff_start++;
    }
    
    // Find last difference
    std::size_t exp_end = expected.size();
    std::size_t act_end = actual.size();
    while (exp_end > diff_start && act_end > diff_start &&
           expected[exp_end - 1] == actual[act_end - 1]) {
        exp_end--;
        act_end--;
    }
    
    return {expected.substr(diff_start, exp_end - diff_start),
            actual.substr(diff_start, act_end - diff_start)};
}

} // namespace scl::test::util

// =============================================================================
// Exception Classes
// =============================================================================

namespace scl::test {

class TestException : public std::exception {
public:
    TestException(const char* file, int line, const std::string& message,
                  const std::string& expected = "", const std::string& actual = "")
        : file_(file), line_(line), message_(message), 
          expected_(expected), actual_(actual) {
        
        std::ostringstream oss;
        oss << file << ":" << line << ": " << message;
        if (!expected.empty() || !actual.empty()) {
            oss << "\n  Expected: " << expected;
            oss << "\n  Actual:   " << actual;
        }
        full_message_ = oss.str();
    }
    
    const char* what() const noexcept override { return full_message_.c_str(); }
    const char* file() const { return file_; }
    int line() const { return line_; }
    const std::string& message() const { return message_; }
    const std::string& expected() const { return expected_; }
    const std::string& actual() const { return actual_; }
    
private:
    const char* file_;
    int line_;
    std::string message_;
    std::string expected_;
    std::string actual_;
    std::string full_message_;
};

class SkipException : public std::exception {
public:
    explicit SkipException(const std::string& reason = "") : reason_(reason) {}
    const char* what() const noexcept override { 
        return reason_.empty() ? "Test skipped" : reason_.c_str(); 
    }
    const std::string& reason() const { return reason_; }
    
private:
    std::string reason_;
};

class TimeoutException : public std::exception {
public:
    explicit TimeoutException(int timeout_ms) : timeout_ms_(timeout_ms) {
        message_ = "Test timed out after " + std::to_string(timeout_ms) + "ms";
    }
    const char* what() const noexcept override { return message_.c_str(); }
    int timeout_ms() const { return timeout_ms_; }
    
private:
    int timeout_ms_;
    std::string message_;
};

class ExpectedFailure : public std::exception {
public:
    explicit ExpectedFailure(const std::string& reason = "") : reason_(reason) {}
    const char* what() const noexcept override { 
        return reason_.empty() ? "Expected failure" : reason_.c_str(); 
    }
    const std::string& reason() const { return reason_; }
    
private:
    std::string reason_;
};

} // namespace scl::test

// =============================================================================
// Test Metadata and Results
// =============================================================================

namespace scl::test {

enum class TestStatus {
    PENDING,
    RUNNING,
    PASSED,
    FAILED,
    SKIPPED,
    XFAIL,      // Expected failure (passed)
    XPASS,      // Unexpected pass (failed)
    ERROR,      // Setup/teardown error
    TIMEOUT
};

inline const char* status_string(TestStatus status) {
    switch (status) {
        case TestStatus::PENDING:  return "PENDING";
        case TestStatus::RUNNING:  return "RUNNING";
        case TestStatus::PASSED:   return "PASSED";
        case TestStatus::FAILED:   return "FAILED";
        case TestStatus::SKIPPED:  return "SKIPPED";
        case TestStatus::XFAIL:    return "XFAIL";
        case TestStatus::XPASS:    return "XPASS";
        case TestStatus::ERROR:    return "ERROR";
        case TestStatus::TIMEOUT:  return "TIMEOUT";
        default:                   return "UNKNOWN";
    }
}

inline const char* status_symbol(TestStatus status, bool use_color = true) {
    if (!use_color) {
        switch (status) {
            case TestStatus::PASSED:   return "[PASS]";
            case TestStatus::FAILED:   return "[FAIL]";
            case TestStatus::SKIPPED:  return "[SKIP]";
            case TestStatus::XFAIL:    return "[XFAIL]";
            case TestStatus::XPASS:    return "[XPASS]";
            case TestStatus::ERROR:    return "[ERROR]";
            case TestStatus::TIMEOUT:  return "[TIMEOUT]";
            default:                   return "[????]";
        }
    }
    
    switch (status) {
        case TestStatus::PASSED:   return "✓";
        case TestStatus::FAILED:   return "✗";
        case TestStatus::SKIPPED:  return "○";
        case TestStatus::XFAIL:    return "✗̶";
        case TestStatus::XPASS:    return "✓!";
        case TestStatus::ERROR:    return "⚠";
        case TestStatus::TIMEOUT:  return "⏱";
        default:                   return "?";
    }
}

using test_func_t = void(*)();
using setup_func_t = void(*)();
using teardown_func_t = void(*)();

struct TestInfo {
    test_func_t func = nullptr;
    const char* name_str = nullptr;   // Renamed to avoid ALL macro parameter conflicts
    const char* file = nullptr;
    int line = 0;
    const char* suite = nullptr;
    std::vector<std::string> tags;
    bool skip = false;
    const char* skip_reason = nullptr;
    bool xfail = false;
    const char* xfail_reason = nullptr;
    int timeout_ms = DEFAULT_TIMEOUT_MS;
    int retry_count = 0;
    setup_func_t setup = nullptr;
    teardown_func_t teardown = nullptr;
};

struct SuiteInfo {
    const char* name = nullptr;
    const char* file = nullptr;
    int line = 0;
    setup_func_t setup = nullptr;
    teardown_func_t teardown = nullptr;
    setup_func_t setup_each = nullptr;
    teardown_func_t teardown_each = nullptr;
    std::vector<std::size_t> test_indices;
};

struct TestResult {
    const TestInfo* test = nullptr;
    TestStatus status = TestStatus::PENDING;
    double duration_ms = 0.0;
    std::string error_message;
    std::string error_file;
    int error_line = 0;
    std::string expected_value;
    std::string actual_value;
    std::string stdout_capture;
    std::string stderr_capture;
    int retry_attempt = 0;
    std::size_t memory_before = 0;
    std::size_t memory_after = 0;
};

} // namespace scl::test

// =============================================================================
// Global Test Storage
// =============================================================================

namespace scl::test::detail {

inline std::array<TestInfo, MAX_TEST_UNITS>& get_tests() {
    static std::array<TestInfo, MAX_TEST_UNITS> tests{};
    return tests;
}

inline std::size_t& get_count() {
    static std::size_t count = 0;
    return count;
}

inline std::array<SuiteInfo, MAX_SUITES>& get_suites() {
    static std::array<SuiteInfo, MAX_SUITES> suites{};
    return suites;
}

inline std::size_t& get_suite_count() {
    static std::size_t count = 0;
    return count;
}

inline const char*& current_suite() {
    static const char* suite = nullptr;
    return suite;
}

inline TestInfo*& current_test_info() {
    static TestInfo* info = nullptr;
    return info;
}

} // namespace scl::test::detail

// =============================================================================
// Test Configuration
// =============================================================================

namespace scl::test {

enum class OutputMode {
    HUMAN,          // Human-readable with colors (default)
    HUMAN_VERBOSE,  // Verbose human-readable
    TAP,            // Test Anything Protocol
    JSON,           // JSON report
    JUNIT_XML,      // JUnit XML (Jenkins/GitLab)
    HTML,           // HTML report
    MARKDOWN,       // Markdown report
    GITHUB,         // GitHub Actions annotations
    TEAMCITY,       // TeamCity service messages
    MINIMAL,        // Dots only
    QUIET           // Errors only
};

enum class Verbosity {
    QUIET = 0,
    NORMAL = 1,
    VERBOSE = 2,
    DEBUG = 3
};

struct Config {
    // Output settings
    OutputMode mode = OutputMode::HUMAN;
    Verbosity verbosity = Verbosity::NORMAL;
    bool color = true;
    bool progress_bar = true;
    bool show_time = true;
    bool show_memory = false;
    
    // Filtering
    const char* filter = nullptr;
    const char* exclude = nullptr;
    std::vector<std::string> tags;
    std::vector<std::string> exclude_tags;
    const char* suite_filter = nullptr;
    
    // Execution control
    bool fail_fast = false;
    bool shuffle = false;
    unsigned int seed = 0;
    int repeat = 1;
    int timeout_ms = DEFAULT_TIMEOUT_MS;
    int retry_count = 0;
    bool stop_on_error = false;
    
    // Output files
    const char* log_file = nullptr;
    const char* json_file = nullptr;
    const char* xml_file = nullptr;
    const char* html_file = nullptr;
    const char* markdown_file = nullptr;
    const char* tap_file = nullptr;
    
    // Advanced
    bool capture_output = false;
    bool show_passed = true;
    bool show_skipped = true;
    bool show_slow = true;
    double slow_threshold_ms = 1000.0;
    bool list_tests = false;
    bool list_tags = false;
    bool dry_run = false;
    int terminal_width = DEFAULT_TERMINAL_WIDTH;
    
    static Config& instance() {
        static Config cfg;
        return cfg;
    }
    
    void auto_detect() {
        if (util::is_tty()) {
            color = true;
            progress_bar = true;
            terminal_width = util::get_terminal_width();
        } else {
            color = false;
            progress_bar = false;
        }
        
        // Check for CI environments
        if (std::getenv("CI") || std::getenv("GITHUB_ACTIONS")) {
            color = true;  // GitHub supports ANSI
            progress_bar = false;
        }
        if (std::getenv("TEAMCITY_VERSION")) {
            mode = OutputMode::TEAMCITY;
            color = false;
        }
        if (std::getenv("GITLAB_CI")) {
            color = true;
        }
    }
};

// =============================================================================
// CLI Parser
// =============================================================================

inline void print_help(const char* prog_name) {
    std::printf(R"(
%sSCL Core - Advanced Test Framework%s

%sUsage:%s %s [options]

%sOutput Formats:%s
  --human               Human-readable output (default)
  --tap                 Test Anything Protocol format
  --json <file>         Export JSON report
  --xml <file>          Export JUnit XML report
  --html <file>         Export HTML report
  --markdown <file>     Export Markdown report
  --github              GitHub Actions annotations format
  --teamcity            TeamCity service messages
  --minimal             Minimal dots output
  --quiet, -q           Show only failures

%sFiltering:%s
  --filter <pattern>    Run tests matching pattern (substring)
  --exclude <pattern>   Exclude tests matching pattern
  --tag <tag>           Run tests with specific tag (can repeat)
  --exclude-tag <tag>   Exclude tests with tag
  --suite <name>        Run specific test suite
  --list                List all tests without running
  --list-tags           List all available tags

%sExecution:%s
  --fail-fast, -x       Stop on first failure
  --shuffle             Randomize test order
  --seed <n>            Random seed for shuffle
  --repeat <n>          Repeat each test n times
  --timeout <ms>        Default test timeout (default: 30000)
  --retry <n>           Retry failed tests n times
  --dry-run             Show what would run without executing

%sOutput Control:%s
  --verbose, -v         Verbose output
  --no-color            Disable ANSI colors
  --no-progress         Disable progress bar
  --no-time             Disable timing info
  --show-all            Show all tests including passed
  --show-slow <ms>      Highlight tests slower than threshold
  --capture             Capture stdout/stderr

%sLogging:%s
  --log <file>          Save detailed log to file
  --tap-file <file>     Save TAP output to file

%sHelp:%s
  --help, -h            Show this help message
  --version             Show version information

%sExamples:%s
  %s%s --filter matrix --verbose%s
  %s%s --tag unit --fail-fast%s
  %s%s --json report.json --xml results.xml%s
  %s%s --github --fail-fast%s
  %s%s --shuffle --seed 42 --repeat 3%s

%sEnvironment Variables:%s
  SCL_TEST_FILTER       Default filter pattern
  SCL_TEST_TIMEOUT      Default timeout in milliseconds
  SCL_TEST_COLOR        Force color (1) or no color (0)
  CI                    Detected for CI-friendly defaults

)",
        color::bold(), color::reset(),
        color::cyan(), color::reset(), prog_name,
        color::yellow(), color::reset(),
        color::yellow(), color::reset(),
        color::yellow(), color::reset(),
        color::yellow(), color::reset(),
        color::yellow(), color::reset(),
        color::yellow(), color::reset(),
        color::yellow(), color::reset(),
        color::dim(), prog_name, color::reset(),
        color::dim(), prog_name, color::reset(),
        color::dim(), prog_name, color::reset(),
        color::dim(), prog_name, color::reset(),
        color::dim(), prog_name, color::reset(),
        color::yellow(), color::reset()
    );
}

inline void print_version() {
    std::printf("SCL Core Test Framework v2.0.0\n");
    std::printf("Built: %s %s\n", __DATE__, __TIME__);
#if defined(__clang__)
    std::printf("Compiler: Clang %d.%d.%d\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    std::printf("Compiler: GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    std::printf("Compiler: MSVC %d\n", _MSC_VER);
#endif
}

inline void parse_args(int argc, char* argv[]) {
    auto& cfg = Config::instance();
    cfg.auto_detect();
    
    // Check environment variables first
    if (const char* env = std::getenv("SCL_TEST_FILTER")) {
        cfg.filter = env;
    }
    if (const char* env = std::getenv("SCL_TEST_TIMEOUT")) {
        cfg.timeout_ms = std::atoi(env);
    }
    if (const char* env = std::getenv("SCL_TEST_COLOR")) {
        cfg.color = (std::strcmp(env, "1") == 0);
    }
    
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        
        // Help and version
        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_help(argv[0]);
            std::exit(0);
        }
        else if (std::strcmp(arg, "--version") == 0) {
            print_version();
            std::exit(0);
        }
        // Output formats
        else if (std::strcmp(arg, "--human") == 0) {
            cfg.mode = OutputMode::HUMAN;
        }
        else if (std::strcmp(arg, "--tap") == 0) {
            cfg.mode = OutputMode::TAP;
        }
        else if (std::strcmp(arg, "--json") == 0 && i + 1 < argc) {
            cfg.json_file = argv[++i];
        }
        else if (std::strcmp(arg, "--xml") == 0 && i + 1 < argc) {
            cfg.xml_file = argv[++i];
        }
        else if (std::strcmp(arg, "--html") == 0 && i + 1 < argc) {
            cfg.html_file = argv[++i];
        }
        else if (std::strcmp(arg, "--markdown") == 0 && i + 1 < argc) {
            cfg.markdown_file = argv[++i];
        }
        else if (std::strcmp(arg, "--github") == 0) {
            cfg.mode = OutputMode::GITHUB;
        }
        else if (std::strcmp(arg, "--teamcity") == 0) {
            cfg.mode = OutputMode::TEAMCITY;
        }
        else if (std::strcmp(arg, "--minimal") == 0) {
            cfg.mode = OutputMode::MINIMAL;
        }
        else if (std::strcmp(arg, "--quiet") == 0 || std::strcmp(arg, "-q") == 0) {
            cfg.mode = OutputMode::QUIET;
            cfg.verbosity = Verbosity::QUIET;
        }
        // Filtering
        else if (std::strcmp(arg, "--filter") == 0 && i + 1 < argc) {
            cfg.filter = argv[++i];
        }
        else if (std::strcmp(arg, "--exclude") == 0 && i + 1 < argc) {
            cfg.exclude = argv[++i];
        }
        else if (std::strcmp(arg, "--tag") == 0 && i + 1 < argc) {
            cfg.tags.push_back(argv[++i]);
        }
        else if (std::strcmp(arg, "--exclude-tag") == 0 && i + 1 < argc) {
            cfg.exclude_tags.push_back(argv[++i]);
        }
        else if (std::strcmp(arg, "--suite") == 0 && i + 1 < argc) {
            cfg.suite_filter = argv[++i];
        }
        else if (std::strcmp(arg, "--list") == 0) {
            cfg.list_tests = true;
        }
        else if (std::strcmp(arg, "--list-tags") == 0) {
            cfg.list_tags = true;
        }
        // Execution control
        else if (std::strcmp(arg, "--fail-fast") == 0 || std::strcmp(arg, "-x") == 0) {
            cfg.fail_fast = true;
        }
        else if (std::strcmp(arg, "--shuffle") == 0) {
            cfg.shuffle = true;
            if (cfg.seed == 0) {
                cfg.seed = static_cast<unsigned int>(std::time(nullptr));
            }
        }
        else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            cfg.seed = static_cast<unsigned int>(std::atoi(argv[++i]));
            cfg.shuffle = true;
        }
        else if (std::strcmp(arg, "--repeat") == 0 && i + 1 < argc) {
            cfg.repeat = std::atoi(argv[++i]);
        }
        else if (std::strcmp(arg, "--timeout") == 0 && i + 1 < argc) {
            cfg.timeout_ms = std::atoi(argv[++i]);
        }
        else if (std::strcmp(arg, "--retry") == 0 && i + 1 < argc) {
            cfg.retry_count = std::atoi(argv[++i]);
        }
        else if (std::strcmp(arg, "--dry-run") == 0) {
            cfg.dry_run = true;
        }
        // Output control
        else if (std::strcmp(arg, "--verbose") == 0 || std::strcmp(arg, "-v") == 0) {
            cfg.verbosity = Verbosity::VERBOSE;
        }
        else if (std::strcmp(arg, "-vv") == 0) {
            cfg.verbosity = Verbosity::DEBUG;
        }
        else if (std::strcmp(arg, "--no-color") == 0) {
            cfg.color = false;
        }
        else if (std::strcmp(arg, "--no-progress") == 0) {
            cfg.progress_bar = false;
        }
        else if (std::strcmp(arg, "--no-time") == 0) {
            cfg.show_time = false;
        }
        else if (std::strcmp(arg, "--show-all") == 0) {
            cfg.show_passed = true;
        }
        else if (std::strcmp(arg, "--show-slow") == 0 && i + 1 < argc) {
            cfg.slow_threshold_ms = std::atof(argv[++i]);
            cfg.show_slow = true;
        }
        else if (std::strcmp(arg, "--capture") == 0) {
            cfg.capture_output = true;
        }
        // Logging
        else if (std::strcmp(arg, "--log") == 0 && i + 1 < argc) {
            cfg.log_file = argv[++i];
        }
        else if (std::strcmp(arg, "--tap-file") == 0 && i + 1 < argc) {
            cfg.tap_file = argv[++i];
        }
        // Unknown
        else if (arg[0] == '-') {
            std::fprintf(stderr, "%sError:%s Unknown option: %s\n", 
                color::error(), color::reset(), arg);
            std::fprintf(stderr, "Use --help for usage information\n");
            std::exit(1);
        }
    }
}

// =============================================================================
// Progress Bar
// =============================================================================

class ProgressBar {
public:
    ProgressBar(int total, int width = 40) 
        : total_(total), width_(width), current_(0), 
          passed_(0), failed_(0), skipped_(0) {}
    
    void update(TestStatus status) {
        current_++;
        switch (status) {
            case TestStatus::PASSED:
            case TestStatus::XFAIL:
                passed_++;
                break;
            case TestStatus::FAILED:
            case TestStatus::ERROR:
            case TestStatus::TIMEOUT:
            case TestStatus::XPASS:
                failed_++;
                break;
            case TestStatus::SKIPPED:
                skipped_++;
                break;
            default:
                break;
        }
        render();
    }
    
    void render() const {
        if (!Config::instance().progress_bar) return;
        
        double progress = static_cast<double>(current_) / total_;
        int filled = static_cast<int>(progress * width_);
        int passed_bar = static_cast<int>((static_cast<double>(passed_) / total_) * width_);
        int failed_bar = static_cast<int>((static_cast<double>(failed_) / total_) * width_);
        
        std::printf("\r");
        
        // Progress bar
        std::printf("%s[%s", color::dim(), color::reset());
        
        for (int i = 0; i < width_; ++i) {
            if (i < passed_bar) {
                std::printf("%s█%s", color::progress_done(), color::reset());
            } else if (i < passed_bar + failed_bar) {
                std::printf("%s█%s", color::progress_fail(), color::reset());
            } else if (i < filled) {
                std::printf("%s█%s", color::dim(), color::reset());
            } else {
                std::printf("%s░%s", color::progress_pending(), color::reset());
            }
        }
        
        std::printf("%s]%s ", color::dim(), color::reset());
        
        // Percentage and counts
        std::printf("%3d%% ", static_cast<int>(progress * 100));
        std::printf("%s%d%s ", color::passed(), passed_, color::reset());
        if (failed_ > 0) {
            std::printf("%s%d%s ", color::failed(), failed_, color::reset());
        }
        if (skipped_ > 0) {
            std::printf("%s%d%s ", color::skipped(), skipped_, color::reset());
        }
        std::printf("/ %d", total_);
        
        std::fflush(stdout);
    }
    
    void finish() const {
        if (!Config::instance().progress_bar) return;
        std::printf("\r%s\r", std::string(80, ' ').c_str());
        std::fflush(stdout);
    }
    
private:
    int total_;
    int width_;
    int current_;
    int passed_;
    int failed_;
    int skipped_;
};

// =============================================================================
// Reporter Classes
// =============================================================================

class Reporter {
public:
    virtual ~Reporter() = default;
    
    virtual void on_run_start(int total_tests, const Config& cfg) = 0;
    virtual void on_test_start(const TestInfo& test) = 0;
    virtual void on_test_end(const TestResult& result) = 0;
    virtual void on_run_end(const std::vector<TestResult>& results, double total_time) = 0;
};

class HumanReporter : public Reporter {
public:
    void on_run_start(int total_tests, const Config& cfg) override {
        auto& c = Config::instance();
        
        std::printf("\n");
        print_header();
        std::printf("  Tests: %s%d%s", color::bold(), total_tests, color::reset());
        if (cfg.filter) {
            std::printf("  Filter: %s%s%s", color::cyan(), cfg.filter, color::reset());
        }
        if (cfg.shuffle) {
            std::printf("  Seed: %s%u%s", color::dim(), cfg.seed, color::reset());
        }
        std::printf("\n");
        print_separator();
        std::printf("\n");
        
        if (c.progress_bar && c.mode == OutputMode::HUMAN) {
            progress_ = std::make_unique<ProgressBar>(total_tests);
        }
    }
    
    void on_test_start(const TestInfo& test) override {
        auto& cfg = Config::instance();
        if (cfg.verbosity >= Verbosity::VERBOSE && cfg.mode == OutputMode::HUMAN_VERBOSE) {
            std::printf("  %sRUNNING%s %s\n", color::blue(), color::reset(), test.name_str);
        }
    }
    
    void on_test_end(const TestResult& result) override {
        auto& cfg = Config::instance();
        
        if (progress_) {
            progress_->update(result.status);
        }
        
        bool should_print = false;
        switch (result.status) {
            case TestStatus::FAILED:
            case TestStatus::ERROR:
            case TestStatus::TIMEOUT:
            case TestStatus::XPASS:
                should_print = true;
                failed_results_.push_back(result);
                break;
            case TestStatus::PASSED:
                should_print = cfg.verbosity >= Verbosity::VERBOSE;
                break;
            case TestStatus::SKIPPED:
                should_print = cfg.show_skipped && cfg.verbosity >= Verbosity::NORMAL;
                break;
            case TestStatus::XFAIL:
                should_print = cfg.verbosity >= Verbosity::VERBOSE;
                break;
            default:
                break;
        }
        
        if (cfg.show_slow && result.duration_ms > cfg.slow_threshold_ms) {
            should_print = true;
        }
        
        if (!progress_ && should_print && cfg.mode != OutputMode::MINIMAL) {
            print_test_result(result);
        }
    }
    
    void on_run_end(const std::vector<TestResult>& results, double total_time) override {
        if (progress_) {
            progress_->finish();
        }
        
        auto& cfg = Config::instance();
        
        // Print failures summary
        if (!failed_results_.empty()) {
            std::printf("\n");
            print_failures_header();
            for (const auto& result : failed_results_) {
                print_failure_detail(result);
            }
        }
        
        // Count results
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        double total_duration = 0;
        
        for (const auto& r : results) {
            total_duration += r.duration_ms;
            switch (r.status) {
                case TestStatus::PASSED:
                case TestStatus::XFAIL:
                    passed++;
                    break;
                case TestStatus::FAILED:
                case TestStatus::XPASS:
                    failed++;
                    break;
                case TestStatus::SKIPPED:
                    skipped++;
                    break;
                case TestStatus::ERROR:
                case TestStatus::TIMEOUT:
                    errors++;
                    break;
                default:
                    break;
            }
        }
        
        // Print summary
        std::printf("\n");
        print_summary_box(results.size(), passed, failed, skipped, errors, total_time);
    }
    
private:
    std::unique_ptr<ProgressBar> progress_;
    std::vector<TestResult> failed_results_;
    
    void print_header() const {
        std::printf("%s", color::bold());
        std::printf("═══════════════════════════════════════════════════════════════════════════════\n");
        std::printf("  %s🧪 SCL Core Test Suite%s\n", color::cyan(), color::reset());
        std::printf("%s", color::bold());
        std::printf("═══════════════════════════════════════════════════════════════════════════════\n");
        std::printf("%s", color::reset());
    }
    
    void print_separator() const {
        std::printf("%s───────────────────────────────────────────────────────────────────────────────%s\n",
            color::dim(), color::reset());
    }
    
    void print_test_result(const TestResult& result) const {
        auto& cfg = Config::instance();
        const char* status_color = "";
        const char* symbol = "";
        
        switch (result.status) {
            case TestStatus::PASSED:
                status_color = color::passed();
                symbol = "✓";
                break;
            case TestStatus::FAILED:
                status_color = color::failed();
                symbol = "✗";
                break;
            case TestStatus::SKIPPED:
                status_color = color::skipped();
                symbol = "○";
                break;
            case TestStatus::XFAIL:
                status_color = color::xfail();
                symbol = "✗";
                break;
            case TestStatus::XPASS:
                status_color = color::xpass();
                symbol = "✓!";
                break;
            case TestStatus::ERROR:
                status_color = color::error();
                symbol = "⚠";
                break;
            case TestStatus::TIMEOUT:
                status_color = color::warning();
                symbol = "⏱";
                break;
            default:
                status_color = color::dim();
                symbol = "?";
                break;
        }
        
        std::printf("  %s%s%s %s", status_color, symbol, color::reset(), result.test->name_str);
        
        if (cfg.show_time && result.duration_ms > 0) {
            if (result.duration_ms > cfg.slow_threshold_ms) {
                std::printf(" %s(%s)%s", color::warning(), 
                    util::format_duration(result.duration_ms).c_str(), color::reset());
            } else {
                std::printf(" %s(%s)%s", color::dim(), 
                    util::format_duration(result.duration_ms).c_str(), color::reset());
            }
        }
        
        if (result.status == TestStatus::SKIPPED && !result.error_message.empty()) {
            std::printf(" %s[%s]%s", color::dim(), result.error_message.c_str(), color::reset());
        }
        
        std::printf("\n");
    }
    
    void print_failures_header() const {
        std::printf("%s%s", color::failed(), color::bold());
        std::printf("═══════════════════════════════════════════════════════════════════════════════\n");
        std::printf("  ❌ FAILURES\n");
        std::printf("═══════════════════════════════════════════════════════════════════════════════\n");
        std::printf("%s", color::reset());
    }
    
    void print_failure_detail(const TestResult& result) const {
        std::printf("\n%s%s── %s ──%s\n", 
            color::failed(), color::bold(), result.test->name_str, color::reset());
        
        // File location
        std::printf("   %sFile:%s %s:%d\n", 
            color::dim(), color::reset(), result.test->file, result.test->line);
        
        // Error message
        if (!result.error_message.empty()) {
            std::printf("\n   %sError:%s\n", color::failed(), color::reset());
            std::printf("   %s%s%s\n", color::bright_red(), result.error_message.c_str(), color::reset());
        }
        
        // Expected vs Actual
        if (!result.expected_value.empty() || !result.actual_value.empty()) {
            std::printf("\n   %sExpected:%s %s%s%s\n", 
                color::dim(), color::reset(), 
                color::green(), result.expected_value.c_str(), color::reset());
            std::printf("   %sActual:  %s %s%s%s\n", 
                color::dim(), color::reset(),
                color::red(), result.actual_value.c_str(), color::reset());
            
            // Show diff if strings differ
            if (!result.expected_value.empty() && !result.actual_value.empty()) {
                auto [exp_diff, act_diff] = util::compute_diff(result.expected_value, result.actual_value);
                if (!exp_diff.empty() || !act_diff.empty()) {
                    std::printf("\n   %sDiff:%s\n", color::dim(), color::reset());
                    std::printf("   %s- %s%s\n", color::red(), exp_diff.c_str(), color::reset());
                    std::printf("   %s+ %s%s\n", color::green(), act_diff.c_str(), color::reset());
                }
            }
        }
        
        // Error location (if different from test location)
        if (!result.error_file.empty() && result.error_line > 0) {
            std::printf("\n   %sAssertion at:%s %s:%d\n",
                color::dim(), color::reset(), result.error_file.c_str(), result.error_line);
        }
        
        std::printf("\n");
    }
    
    void print_summary_box(int total, int passed, int failed, int skipped, int errors, double total_time) const {
        auto& cfg = Config::instance();
        
        bool all_passed = (failed == 0 && errors == 0);
        
        // Box top
        if (all_passed) {
            std::printf("%s%s", color::passed(), color::bold());
        } else {
            std::printf("%s%s", color::failed(), color::bold());
        }
        std::printf("═══════════════════════════════════════════════════════════════════════════════\n");
        
        if (all_passed) {
            std::printf("  ✅ All tests passed!\n");
        } else {
            std::printf("  ❌ %d test(s) failed\n", failed + errors);
        }
        
        std::printf("═══════════════════════════════════════════════════════════════════════════════%s\n", 
            color::reset());
        
        // Stats line (pytest-style)
        std::printf("\n  ");
        
        if (passed > 0) {
            std::printf("%s%d passed%s", color::passed(), passed, color::reset());
        }
        if (failed > 0) {
            if (passed > 0) std::printf(", ");
            std::printf("%s%d failed%s", color::failed(), failed, color::reset());
        }
        if (errors > 0) {
            if (passed > 0 || failed > 0) std::printf(", ");
            std::printf("%s%d error%s", color::error(), errors, color::reset());
        }
        if (skipped > 0) {
            if (passed > 0 || failed > 0 || errors > 0) std::printf(", ");
            std::printf("%s%d skipped%s", color::skipped(), skipped, color::reset());
        }
        
        if (cfg.show_time) {
            std::printf(" %sin %s%s", color::dim(), 
                util::format_duration(total_time * 1000).c_str(), color::reset());
        }
        
        std::printf("\n\n");
    }
};

class TAPReporter : public Reporter {
public:
    explicit TAPReporter(FILE* output = stdout) : output_(output) {}
    
    void on_run_start(int total_tests, const Config&) override {
        std::fprintf(output_, "TAP version 14\n");
        std::fprintf(output_, "1..%d\n", total_tests);
    }
    
    void on_test_start(const TestInfo&) override {}
    
    void on_test_end(const TestResult& result) override {
        test_num_++;
        
        const char* status = (result.status == TestStatus::PASSED || 
                              result.status == TestStatus::XFAIL) ? "ok" : "not ok";
        
        std::fprintf(output_, "%s %d - %s", status, test_num_, result.test->name_str);
        
        if (result.status == TestStatus::SKIPPED) {
            std::fprintf(output_, " # SKIP %s", result.error_message.c_str());
        } else if (result.status == TestStatus::XFAIL) {
            std::fprintf(output_, " # TODO expected failure");
        }
        
        std::fprintf(output_, "\n");
        
        if (result.status == TestStatus::FAILED || result.status == TestStatus::ERROR) {
            std::fprintf(output_, "  ---\n");
            std::fprintf(output_, "  message: '%s'\n", 
                util::escape_json(result.error_message).c_str());
            std::fprintf(output_, "  severity: fail\n");
            std::fprintf(output_, "  at:\n");
            std::fprintf(output_, "    file: '%s'\n", result.test->file);
            std::fprintf(output_, "    line: %d\n", result.test->line);
            if (!result.expected_value.empty()) {
                std::fprintf(output_, "  expected: '%s'\n", 
                    util::escape_json(result.expected_value).c_str());
            }
            if (!result.actual_value.empty()) {
                std::fprintf(output_, "  actual: '%s'\n", 
                    util::escape_json(result.actual_value).c_str());
            }
            std::fprintf(output_, "  duration_ms: %.3f\n", result.duration_ms);
            std::fprintf(output_, "  ...\n");
        }
    }
    
    void on_run_end(const std::vector<TestResult>&, double) override {
        // TAP format complete
    }
    
private:
    FILE* output_;
    int test_num_ = 0;
};

class MinimalReporter : public Reporter {
public:
    void on_run_start(int, const Config&) override {}
    void on_test_start(const TestInfo&) override {}
    
    void on_test_end(const TestResult& result) override {
        switch (result.status) {
            case TestStatus::PASSED:
            case TestStatus::XFAIL:
                std::printf("%s.%s", color::passed(), color::reset());
                break;
            case TestStatus::FAILED:
            case TestStatus::XPASS:
                std::printf("%sF%s", color::failed(), color::reset());
                failed_results_.push_back(result);
                break;
            case TestStatus::ERROR:
            case TestStatus::TIMEOUT:
                std::printf("%sE%s", color::error(), color::reset());
                failed_results_.push_back(result);
                break;
            case TestStatus::SKIPPED:
                std::printf("%ss%s", color::skipped(), color::reset());
                break;
            default:
                std::printf("?");
                break;
        }
        std::fflush(stdout);
    }
    
    void on_run_end(const std::vector<TestResult>& results, double total_time) override {
        std::printf("\n\n");
        
        // Print failure details
        if (!failed_results_.empty()) {
            for (const auto& result : failed_results_) {
                std::printf("%s%s%s %s:%d\n  %s\n\n",
                    color::failed(), result.test->name_str, color::reset(),
                    result.test->file, result.test->line,
                    result.error_message.c_str());
            }
        }
        
        int passed = 0, failed = 0;
        for (const auto& r : results) {
            if (r.status == TestStatus::PASSED || r.status == TestStatus::XFAIL) passed++;
            else if (r.status == TestStatus::FAILED || r.status == TestStatus::ERROR || 
                     r.status == TestStatus::TIMEOUT || r.status == TestStatus::XPASS) failed++;
        }
        
        std::printf("%d/%d passed", passed, static_cast<int>(results.size()));
        if (failed > 0) std::printf(", %s%d failed%s", color::failed(), failed, color::reset());
        std::printf(" (%.2fs)\n", total_time);
    }
    
private:
    std::vector<TestResult> failed_results_;
};

class GitHubReporter : public Reporter {
public:
    void on_run_start(int total_tests, const Config&) override {
        std::printf("::group::SCL Test Suite (%d tests)\n", total_tests);
    }
    
    void on_test_start(const TestInfo&) override {}
    
    void on_test_end(const TestResult& result) override {
        if (result.status == TestStatus::FAILED || result.status == TestStatus::ERROR) {
            std::printf("::error file=%s,line=%d::%s - %s\n",
                result.test->file, result.test->line,
                result.test->name_str,
                util::escape_json(result.error_message).c_str());
        } else if (result.status == TestStatus::SKIPPED) {
            std::printf("::warning file=%s,line=%d::%s - skipped\n",
                result.test->file, result.test->line,
                result.test->name_str);
        }
    }
    
    void on_run_end(const std::vector<TestResult>& results, double total_time) override {
        int passed = 0, failed = 0, skipped = 0;
        for (const auto& r : results) {
            switch (r.status) {
                case TestStatus::PASSED: case TestStatus::XFAIL: passed++; break;
                case TestStatus::FAILED: case TestStatus::ERROR: case TestStatus::TIMEOUT: failed++; break;
                case TestStatus::SKIPPED: skipped++; break;
                default: break;
            }
        }
        
        std::printf("::endgroup::\n");
        
        if (failed > 0) {
            std::printf("::error::Tests failed: %d passed, %d failed, %d skipped (%.2fs)\n",
                passed, failed, skipped, total_time);
        } else {
            std::printf("::notice::Tests passed: %d passed, %d skipped (%.2fs)\n",
                passed, skipped, total_time);
        }
    }
};

class TeamCityReporter : public Reporter {
public:
    void on_run_start(int, const Config&) override {
        std::printf("##teamcity[testSuiteStarted name='SCL Test Suite']\n");
    }
    
    void on_test_start(const TestInfo& test) override {
        std::printf("##teamcity[testStarted name='%s']\n", escape_tc(test.name_str).c_str());
    }
    
    void on_test_end(const TestResult& result) override {
        std::string name = escape_tc(result.test->name_str);
        
        switch (result.status) {
            case TestStatus::FAILED:
            case TestStatus::ERROR:
            case TestStatus::TIMEOUT:
                std::printf("##teamcity[testFailed name='%s' message='%s' details='%s:%d']\n",
                    name.c_str(),
                    escape_tc(result.error_message).c_str(),
                    result.test->file, result.test->line);
                break;
            case TestStatus::SKIPPED:
                std::printf("##teamcity[testIgnored name='%s' message='%s']\n",
                    name.c_str(),
                    escape_tc(result.error_message).c_str());
                break;
            default:
                break;
        }
        
        std::printf("##teamcity[testFinished name='%s' duration='%d']\n",
            name.c_str(), static_cast<int>(result.duration_ms));
    }
    
    void on_run_end(const std::vector<TestResult>&, double) override {
        std::printf("##teamcity[testSuiteFinished name='SCL Test Suite']\n");
    }
    
private:
    static std::string escape_tc(const std::string& s) {
        std::string result;
        for (char c : s) {
            switch (c) {
                case '|': result += "||"; break;
                case '\'': result += "|'"; break;
                case '\n': result += "|n"; break;
                case '\r': result += "|r"; break;
                case '[': result += "|["; break;
                case ']': result += "|]"; break;
                default: result += c; break;
            }
        }
        return result;
    }
};

// =============================================================================
// File Exporters
// =============================================================================

class FileExporter {
public:
    static void export_json(const std::vector<TestResult>& results, double total_time, 
                           const char* filename) {
        FILE* f = std::fopen(filename, "w");
        if (!f) {
            std::fprintf(stderr, "Failed to open JSON file: %s\n", filename);
            return;
        }
        
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        for (const auto& r : results) {
            switch (r.status) {
                case TestStatus::PASSED: case TestStatus::XFAIL: passed++; break;
                case TestStatus::FAILED: case TestStatus::XPASS: failed++; break;
                case TestStatus::SKIPPED: skipped++; break;
                case TestStatus::ERROR: case TestStatus::TIMEOUT: errors++; break;
                default: break;
            }
        }
        
        std::fprintf(f, "{\n");
        std::fprintf(f, "  \"timestamp\": \"%s\",\n", util::timestamp_iso8601().c_str());
        std::fprintf(f, "  \"duration\": %.3f,\n", total_time);
        std::fprintf(f, "  \"summary\": {\n");
        std::fprintf(f, "    \"total\": %zu,\n", results.size());
        std::fprintf(f, "    \"passed\": %d,\n", passed);
        std::fprintf(f, "    \"failed\": %d,\n", failed);
        std::fprintf(f, "    \"skipped\": %d,\n", skipped);
        std::fprintf(f, "    \"errors\": %d\n", errors);
        std::fprintf(f, "  },\n");
        std::fprintf(f, "  \"tests\": [\n");
        
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::fprintf(f, "    {\n");
            std::fprintf(f, "      \"name\": \"%s\",\n", util::escape_json(r.test->name_str).c_str());
            std::fprintf(f, "      \"file\": \"%s\",\n", util::escape_json(r.test->file).c_str());
            std::fprintf(f, "      \"line\": %d,\n", r.test->line);
            std::fprintf(f, "      \"status\": \"%s\",\n", status_string(r.status));
            std::fprintf(f, "      \"duration_ms\": %.3f", r.duration_ms);
            if (!r.error_message.empty()) {
                std::fprintf(f, ",\n      \"error\": \"%s\"", 
                    util::escape_json(r.error_message).c_str());
            }
            if (!r.expected_value.empty()) {
                std::fprintf(f, ",\n      \"expected\": \"%s\"", 
                    util::escape_json(r.expected_value).c_str());
            }
            if (!r.actual_value.empty()) {
                std::fprintf(f, ",\n      \"actual\": \"%s\"", 
                    util::escape_json(r.actual_value).c_str());
            }
            std::fprintf(f, "\n    }%s\n", i < results.size() - 1 ? "," : "");
        }
        
        std::fprintf(f, "  ]\n");
        std::fprintf(f, "}\n");
        
        std::fclose(f);
    }
    
    static void export_junit_xml(const std::vector<TestResult>& results, double total_time,
                                 const char* filename) {
        FILE* f = std::fopen(filename, "w");
        if (!f) {
            std::fprintf(stderr, "Failed to open XML file: %s\n", filename);
            return;
        }
        
        int failures = 0, errors = 0, skipped = 0;
        for (const auto& r : results) {
            switch (r.status) {
                case TestStatus::FAILED: case TestStatus::XPASS: failures++; break;
                case TestStatus::ERROR: case TestStatus::TIMEOUT: errors++; break;
                case TestStatus::SKIPPED: skipped++; break;
                default: break;
            }
        }
        
        std::fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        std::fprintf(f, "<testsuites name=\"SCL Test Suite\" tests=\"%zu\" failures=\"%d\" "
                       "errors=\"%d\" skipped=\"%d\" time=\"%.3f\" timestamp=\"%s\">\n",
                    results.size(), failures, errors, skipped, total_time,
                    util::timestamp_iso8601().c_str());
        
        // Group by suite
        std::map<std::string, std::vector<const TestResult*>> by_suite;
        for (const auto& r : results) {
            std::string suite = r.test->suite ? r.test->suite : "default";
            by_suite[suite].push_back(&r);
        }
        
        for (const auto& [suite_name, suite_results] : by_suite) {
            int suite_failures = 0, suite_errors = 0, suite_skipped = 0;
            double suite_time = 0;
            for (const auto* r : suite_results) {
                suite_time += r->duration_ms / 1000.0;
                switch (r->status) {
                    case TestStatus::FAILED: case TestStatus::XPASS: suite_failures++; break;
                    case TestStatus::ERROR: case TestStatus::TIMEOUT: suite_errors++; break;
                    case TestStatus::SKIPPED: suite_skipped++; break;
                    default: break;
                }
            }
            
            std::fprintf(f, "  <testsuite name=\"%s\" tests=\"%zu\" failures=\"%d\" "
                           "errors=\"%d\" skipped=\"%d\" time=\"%.3f\">\n",
                        util::escape_xml(suite_name).c_str(), suite_results.size(),
                        suite_failures, suite_errors, suite_skipped, suite_time);
            
            for (const auto* r : suite_results) {
                std::fprintf(f, "    <testcase name=\"%s\" classname=\"%s\" time=\"%.3f\"",
                    util::escape_xml(r->test->name_str).c_str(),
                    util::escape_xml(suite_name).c_str(),
                    r->duration_ms / 1000.0);
                
                if (r->status == TestStatus::PASSED || r->status == TestStatus::XFAIL) {
                    std::fprintf(f, "/>\n");
                } else {
                    std::fprintf(f, ">\n");
                    
                    switch (r->status) {
                        case TestStatus::FAILED:
                        case TestStatus::XPASS:
                            std::fprintf(f, "      <failure message=\"%s\" type=\"AssertionError\">\n",
                                util::escape_xml(r->error_message).c_str());
                            std::fprintf(f, "%s:%d\n", r->test->file, r->test->line);
                            if (!r->expected_value.empty()) {
                                std::fprintf(f, "Expected: %s\n", 
                                    util::escape_xml(r->expected_value).c_str());
                            }
                            if (!r->actual_value.empty()) {
                                std::fprintf(f, "Actual: %s\n", 
                                    util::escape_xml(r->actual_value).c_str());
                            }
                            std::fprintf(f, "      </failure>\n");
                            break;
                        case TestStatus::ERROR:
                        case TestStatus::TIMEOUT:
                            std::fprintf(f, "      <error message=\"%s\" type=\"%s\">\n",
                                util::escape_xml(r->error_message).c_str(),
                                r->status == TestStatus::TIMEOUT ? "Timeout" : "Error");
                            std::fprintf(f, "%s:%d\n", r->test->file, r->test->line);
                            std::fprintf(f, "      </error>\n");
                            break;
                        case TestStatus::SKIPPED:
                            std::fprintf(f, "      <skipped message=\"%s\"/>\n",
                                util::escape_xml(r->error_message).c_str());
                            break;
                        default:
                            break;
                    }
                    
                    std::fprintf(f, "    </testcase>\n");
                }
            }
            
            std::fprintf(f, "  </testsuite>\n");
        }
        
        std::fprintf(f, "</testsuites>\n");
        std::fclose(f);
    }
    
    static void export_html(const std::vector<TestResult>& results, double total_time,
                           const char* filename) {
        FILE* f = std::fopen(filename, "w");
        if (!f) {
            std::fprintf(stderr, "Failed to open HTML file: %s\n", filename);
            return;
        }
        
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        for (const auto& r : results) {
            switch (r.status) {
                case TestStatus::PASSED: case TestStatus::XFAIL: passed++; break;
                case TestStatus::FAILED: case TestStatus::XPASS: failed++; break;
                case TestStatus::SKIPPED: skipped++; break;
                case TestStatus::ERROR: case TestStatus::TIMEOUT: errors++; break;
                default: break;
            }
        }
        
        std::fprintf(f, R"html(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCL Test Report</title>
    <style>
        :root {
            --bg-primary: #1a1b26;
            --bg-secondary: #24283b;
            --bg-tertiary: #414868;
            --text-primary: #c0caf5;
            --text-secondary: #a9b1d6;
            --text-muted: #565f89;
            --accent-green: #9ece6a;
            --accent-red: #f7768e;
            --accent-yellow: #e0af68;
            --accent-blue: #7aa2f7;
            --accent-purple: #bb9af7;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: var(--accent-blue);
        }
        .timestamp {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .stat {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .stat.passed .stat-value { color: var(--accent-green); }
        .stat.failed .stat-value { color: var(--accent-red); }
        .stat.skipped .stat-value { color: var(--accent-yellow); }
        .stat.total .stat-value { color: var(--accent-blue); }
        .test-list { margin-top: 2rem; }
        .test-item {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .test-item:hover { background: var(--bg-tertiary); }
        .test-status {
            width: 24px;
            height: 24px;
            border-radius: 50%%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .test-status.passed { background: var(--accent-green); color: var(--bg-primary); }
        .test-status.failed { background: var(--accent-red); color: var(--bg-primary); }
        .test-status.skipped { background: var(--accent-yellow); color: var(--bg-primary); }
        .test-status.error { background: var(--accent-purple); color: var(--bg-primary); }
        .test-name { flex: 1; font-weight: 500; }
        .test-duration { color: var(--text-muted); font-size: 0.85rem; }
        .test-file { color: var(--text-muted); font-size: 0.8rem; }
        .test-error {
            background: rgba(247, 118, 142, 0.1);
            border-left: 3px solid var(--accent-red);
            padding: 1rem;
            margin-top: 0.5rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
        }
        .filter-bar {
            background: var(--bg-secondary);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            display: flex;
            gap: 0.5rem;
        }
        .filter-btn {
            background: var(--bg-tertiary);
            border: none;
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
        }
        .filter-btn:hover { background: var(--accent-blue); color: var(--bg-primary); }
        .filter-btn.active { background: var(--accent-blue); color: var(--bg-primary); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 SCL Test Report</h1>
        <p class="timestamp">Generated: %s | Duration: %.2fs</p>
        
        <div class="summary">
            <div class="stat total">
                <div class="stat-value">%zu</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat passed">
                <div class="stat-value">%d</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat failed">
                <div class="stat-value">%d</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat skipped">
                <div class="stat-value">%d</div>
                <div class="stat-label">Skipped</div>
            </div>
        </div>
        
        <div class="filter-bar">
            <button class="filter-btn active" onclick="filterTests('all')">All</button>
            <button class="filter-btn" onclick="filterTests('passed')">Passed</button>
            <button class="filter-btn" onclick="filterTests('failed')">Failed</button>
            <button class="filter-btn" onclick="filterTests('skipped')">Skipped</button>
        </div>
        
        <div class="test-list">
)html",
            util::timestamp_iso8601().c_str(), total_time,
            results.size(), passed, failed + errors, skipped);
        
        for (const auto& r : results) {
            const char* status_class = "passed";
            const char* status_symbol = "✓";
            
            switch (r.status) {
                case TestStatus::FAILED: case TestStatus::XPASS:
                    status_class = "failed"; status_symbol = "✗"; break;
                case TestStatus::ERROR: case TestStatus::TIMEOUT:
                    status_class = "error"; status_symbol = "!"; break;
                case TestStatus::SKIPPED:
                    status_class = "skipped"; status_symbol = "○"; break;
                default: break;
            }
            
            std::fprintf(f, R"(            <div class="test-item" data-status="%s">
                <div class="test-status %s">%s</div>
                <div class="test-name">%s</div>
                <div class="test-duration">%.2fms</div>
                <div class="test-file">%s:%d</div>
)",
                status_class, status_class, status_symbol,
                util::escape_html(r.test->name_str).c_str(),
                r.duration_ms,
                util::escape_html(r.test->file).c_str(),
                r.test->line);
            
            if (!r.error_message.empty()) {
                std::fprintf(f, R"(                <div class="test-error">%s</div>
)",
                    util::escape_html(r.error_message).c_str());
            }
            
            std::fprintf(f, "            </div>\n");
        }
        
        std::fprintf(f, R"(        </div>
    </div>
    <script>
        function filterTests(status) {
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            document.querySelectorAll('.test-item').forEach(item => {
                if (status === 'all' || item.dataset.status === status) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
)");
        
        std::fclose(f);
    }
    
    static void export_markdown(const std::vector<TestResult>& results, double total_time,
                               const char* filename) {
        FILE* f = std::fopen(filename, "w");
        if (!f) {
            std::fprintf(stderr, "Failed to open Markdown file: %s\n", filename);
            return;
        }
        
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        for (const auto& r : results) {
            switch (r.status) {
                case TestStatus::PASSED: case TestStatus::XFAIL: passed++; break;
                case TestStatus::FAILED: case TestStatus::XPASS: failed++; break;
                case TestStatus::SKIPPED: skipped++; break;
                case TestStatus::ERROR: case TestStatus::TIMEOUT: errors++; break;
                default: break;
            }
        }
        
        std::fprintf(f, "# 🧪 SCL Test Report\n\n");
        std::fprintf(f, "**Generated:** %s  \n", util::timestamp_iso8601().c_str());
        std::fprintf(f, "**Duration:** %.2fs\n\n", total_time);
        
        std::fprintf(f, "## Summary\n\n");
        std::fprintf(f, "| Status | Count |\n");
        std::fprintf(f, "|--------|-------|\n");
        std::fprintf(f, "| ✅ Passed | %d |\n", passed);
        std::fprintf(f, "| ❌ Failed | %d |\n", failed + errors);
        std::fprintf(f, "| ⏭️ Skipped | %d |\n", skipped);
        std::fprintf(f, "| **Total** | **%zu** |\n\n", results.size());
        
        // Failures first
        bool has_failures = false;
        for (const auto& r : results) {
            if (r.status == TestStatus::FAILED || r.status == TestStatus::ERROR ||
                r.status == TestStatus::TIMEOUT || r.status == TestStatus::XPASS) {
                if (!has_failures) {
                    std::fprintf(f, "## ❌ Failures\n\n");
                    has_failures = true;
                }
                
                std::fprintf(f, "### `%s`\n\n", r.test->name_str);
                std::fprintf(f, "**File:** `%s:%d`  \n", r.test->file, r.test->line);
                std::fprintf(f, "**Duration:** %.2fms\n\n", r.duration_ms);
                
                if (!r.error_message.empty()) {
                    std::fprintf(f, "```\n%s\n```\n\n", r.error_message.c_str());
                }
                
                if (!r.expected_value.empty() || !r.actual_value.empty()) {
                    std::fprintf(f, "| | Value |\n");
                    std::fprintf(f, "|---|---|\n");
                    if (!r.expected_value.empty()) {
                        std::fprintf(f, "| Expected | `%s` |\n", r.expected_value.c_str());
                    }
                    if (!r.actual_value.empty()) {
                        std::fprintf(f, "| Actual | `%s` |\n", r.actual_value.c_str());
                    }
                    std::fprintf(f, "\n");
                }
            }
        }
        
        // All tests table
        std::fprintf(f, "## All Tests\n\n");
        std::fprintf(f, "| Status | Test | Duration | File |\n");
        std::fprintf(f, "|--------|------|----------|------|\n");
        
        for (const auto& r : results) {
            const char* status_emoji = "✅";
            switch (r.status) {
                case TestStatus::FAILED: case TestStatus::XPASS: status_emoji = "❌"; break;
                case TestStatus::ERROR: case TestStatus::TIMEOUT: status_emoji = "⚠️"; break;
                case TestStatus::SKIPPED: status_emoji = "⏭️"; break;
                default: break;
            }
            
            std::fprintf(f, "| %s | `%s` | %.2fms | `%s:%d` |\n",
                status_emoji, r.test->name_str, r.duration_ms, r.test->file, r.test->line);
        }
        
        std::fclose(f);
    }
};

// =============================================================================
// Test Runner
// =============================================================================

class Runner {
public:
    Runner() : cfg_(Config::instance()) {
        // Open log file if specified
        if (cfg_.log_file) {
            log_ = std::fopen(cfg_.log_file, "w");
            if (!log_) {
                std::fprintf(stderr, "Warning: Failed to open log file: %s\n", cfg_.log_file);
            }
        }
        
        // Create appropriate reporter
        switch (cfg_.mode) {
            case OutputMode::TAP:
                reporter_ = std::make_unique<TAPReporter>();
                break;
            case OutputMode::MINIMAL:
                reporter_ = std::make_unique<MinimalReporter>();
                break;
            case OutputMode::GITHUB:
                reporter_ = std::make_unique<GitHubReporter>();
                break;
            case OutputMode::TEAMCITY:
                reporter_ = std::make_unique<TeamCityReporter>();
                break;
            case OutputMode::QUIET:
                reporter_ = std::make_unique<MinimalReporter>();
                break;
            default:
                reporter_ = std::make_unique<HumanReporter>();
                break;
        }
    }
    
    ~Runner() {
        if (log_) {
            std::fclose(log_);
        }
    }
    
    int run() {
        // Handle list modes
        if (cfg_.list_tests) {
            return list_tests();
        }
        if (cfg_.list_tags) {
            return list_tags();
        }
        
        // Build test list
        std::vector<std::size_t> test_indices;
        const auto& tests = detail::get_tests();
        const std::size_t count = detail::get_count();
        
        for (std::size_t i = 0; i < count; ++i) {
            if (!tests[i].func) continue;
            if (!should_run(tests[i])) continue;
            test_indices.push_back(i);
        }
        
        // Shuffle if requested
        if (cfg_.shuffle && !test_indices.empty()) {
            std::mt19937 rng(cfg_.seed);
            std::shuffle(test_indices.begin(), test_indices.end(), rng);
        }
        
        // Dry run
        if (cfg_.dry_run) {
            return dry_run(test_indices);
        }
        
        // Run tests
        int total = static_cast<int>(test_indices.size()) * cfg_.repeat;
        reporter_->on_run_start(total, cfg_);
        
        auto start_time = std::chrono::steady_clock::now();
        
        for (int repeat = 0; repeat < cfg_.repeat; ++repeat) {
            for (std::size_t idx : test_indices) {
                auto result = run_test(tests[idx]);
                results_.push_back(result);
                
                reporter_->on_test_end(result);
                
                if (cfg_.fail_fast && is_failure(result.status)) {
                    break;
                }
            }
            
            if (cfg_.fail_fast && !results_.empty() && is_failure(results_.back().status)) {
                break;
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        reporter_->on_run_end(results_, total_time);
        
        // Export files
        if (cfg_.json_file) {
            FileExporter::export_json(results_, total_time, cfg_.json_file);
        }
        if (cfg_.xml_file) {
            FileExporter::export_junit_xml(results_, total_time, cfg_.xml_file);
        }
        if (cfg_.html_file) {
            FileExporter::export_html(results_, total_time, cfg_.html_file);
        }
        if (cfg_.markdown_file) {
            FileExporter::export_markdown(results_, total_time, cfg_.markdown_file);
        }
        if (cfg_.tap_file) {
            FILE* tap = std::fopen(cfg_.tap_file, "w");
            if (tap) {
                TAPReporter tap_reporter(tap);
                tap_reporter.on_run_start(static_cast<int>(results_.size()), cfg_);
                for (const auto& r : results_) {
                    tap_reporter.on_test_end(r);
                }
                tap_reporter.on_run_end(results_, total_time);
                std::fclose(tap);
            }
        }
        
        // Return exit code
        for (const auto& r : results_) {
            if (is_failure(r.status)) {
                return 1;
            }
        }
        return 0;
    }
    
private:
    const Config& cfg_;
    FILE* log_ = nullptr;
    std::unique_ptr<Reporter> reporter_;
    std::vector<TestResult> results_;
    
    static bool is_failure(TestStatus status) {
        return status == TestStatus::FAILED || 
               status == TestStatus::ERROR ||
               status == TestStatus::TIMEOUT ||
               status == TestStatus::XPASS;
    }
    
    bool should_run(const TestInfo& test) const {
        // Filter by name
        if (cfg_.filter && std::strstr(test.name_str, cfg_.filter) == nullptr) {
            return false;
        }
        
        // Exclude by name
        if (cfg_.exclude && std::strstr(test.name_str, cfg_.exclude) != nullptr) {
            return false;
        }
        
        // Filter by suite
        if (cfg_.suite_filter) {
            if (!test.suite || std::strstr(test.suite, cfg_.suite_filter) == nullptr) {
                return false;
            }
        }
        
        // Filter by tags
        if (!cfg_.tags.empty()) {
            bool has_tag = false;
            for (const auto& tag : cfg_.tags) {
                for (const auto& test_tag : test.tags) {
                    if (test_tag == tag) {
                        has_tag = true;
                        break;
                    }
                }
                if (has_tag) break;
            }
            if (!has_tag) return false;
        }
        
        // Exclude by tags
        for (const auto& exclude_tag : cfg_.exclude_tags) {
            for (const auto& test_tag : test.tags) {
                if (test_tag == exclude_tag) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    TestResult run_test(const TestInfo& test) {
        TestResult result;
        result.test = &test;
        
        // Check if test should be skipped
        if (test.skip) {
            result.status = TestStatus::SKIPPED;
            result.error_message = test.skip_reason ? test.skip_reason : "Skipped";
            return result;
        }
        
        reporter_->on_test_start(test);
        
        // Run with retries
        int max_retries = std::max(test.retry_count, cfg_.retry_count);
        
        for (int attempt = 0; attempt <= max_retries; ++attempt) {
            result.retry_attempt = attempt;
            
            auto t0 = std::chrono::steady_clock::now();
            
            try {
                // Run setup
                if (test.setup) {
                    test.setup();
                }
                
                // Run test
                test.func();
                
                // Run teardown
                if (test.teardown) {
                    test.teardown();
                }
                
                auto t1 = std::chrono::steady_clock::now();
                result.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                
                // Handle expected failure
                if (test.xfail) {
                    result.status = TestStatus::XPASS;
                    result.error_message = "Test passed unexpectedly";
                } else {
                    result.status = TestStatus::PASSED;
                }
                
                break; // Success, no retry needed
                
            } catch (const SkipException& e) {
                auto t1 = std::chrono::steady_clock::now();
                result.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                result.status = TestStatus::SKIPPED;
                result.error_message = e.reason();
                break;
                
            } catch (const TestException& e) {
                auto t1 = std::chrono::steady_clock::now();
                result.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                
                if (test.xfail) {
                    result.status = TestStatus::XFAIL;
                } else {
                    result.status = TestStatus::FAILED;
                }
                
                result.error_message = e.message();
                result.error_file = e.file();
                result.error_line = e.line();
                result.expected_value = e.expected();
                result.actual_value = e.actual();
                
                if (attempt == max_retries) break;
                
            } catch (const TimeoutException& e) {
                auto t1 = std::chrono::steady_clock::now();
                result.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                result.status = TestStatus::TIMEOUT;
                result.error_message = e.what();
                break;
                
            } catch (const std::exception& e) {
                auto t1 = std::chrono::steady_clock::now();
                result.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                result.status = TestStatus::ERROR;
                result.error_message = std::string("Uncaught exception: ") + e.what();
                
                if (attempt == max_retries) break;
                
            } catch (...) {
                auto t1 = std::chrono::steady_clock::now();
                result.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                result.status = TestStatus::ERROR;
                result.error_message = "Unknown exception";
                
                if (attempt == max_retries) break;
            }
            
            // Log retry
            if (log_ && attempt < max_retries) {
                std::fprintf(log_, "[RETRY] %s (attempt %d/%d)\n", 
                    test.name_str, attempt + 1, max_retries + 1);
            }
        }
        
        // Log result
        if (log_) {
            std::fprintf(log_, "[%s] %s (%.2fms)\n",
                status_string(result.status), test.name_str, result.duration_ms);
            if (!result.error_message.empty()) {
                std::fprintf(log_, "  Error: %s\n", result.error_message.c_str());
            }
        }
        
        return result;
    }
    
    int list_tests() const {
        const auto& tests = detail::get_tests();
        const std::size_t count = detail::get_count();
        
        std::printf("\n%sAvailable tests:%s\n\n", color::bold(), color::reset());
        
        int num = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (!tests[i].func) continue;
            if (!should_run(tests[i])) continue;
            
            num++;
            std::printf("  %s%s%s", color::cyan(), tests[i].name_str, color::reset());
            
            if (tests[i].suite) {
                std::printf(" %s[%s]%s", color::dim(), tests[i].suite, color::reset());
            }
            
            if (!tests[i].tags.empty()) {
                std::printf(" %s(", color::dim());
                for (std::size_t t = 0; t < tests[i].tags.size(); ++t) {
                    if (t > 0) std::printf(", ");
                    std::printf("%s", tests[i].tags[t].c_str());
                }
                std::printf(")%s", color::reset());
            }
            
            if (tests[i].skip) {
                std::printf(" %s[SKIP]%s", color::skipped(), color::reset());
            }
            if (tests[i].xfail) {
                std::printf(" %s[XFAIL]%s", color::xfail(), color::reset());
            }
            
            std::printf("\n");
            std::printf("    %s%s:%d%s\n", color::dim(), tests[i].file, tests[i].line, color::reset());
        }
        
        std::printf("\n%s%d test(s) total%s\n\n", color::dim(), num, color::reset());
        return 0;
    }
    
    int list_tags() const {
        const auto& tests = detail::get_tests();
        const std::size_t count = detail::get_count();
        
        std::set<std::string> all_tags;
        std::map<std::string, int> tag_counts;
        
        for (std::size_t i = 0; i < count; ++i) {
            if (!tests[i].func) continue;
            for (const auto& tag : tests[i].tags) {
                all_tags.insert(tag);
                tag_counts[tag]++;
            }
        }
        
        std::printf("\n%sAvailable tags:%s\n\n", color::bold(), color::reset());
        
        for (const auto& tag : all_tags) {
            std::printf("  %s%s%s %s(%d tests)%s\n",
                color::cyan(), tag.c_str(), color::reset(),
                color::dim(), tag_counts[tag], color::reset());
        }
        
        std::printf("\n%s%zu tag(s) total%s\n\n", color::dim(), all_tags.size(), color::reset());
        return 0;
    }
    
    int dry_run(const std::vector<std::size_t>& test_indices) const {
        const auto& tests = detail::get_tests();
        
        std::printf("\n%sDry run - would execute:%s\n\n", color::bold(), color::reset());
        
        for (std::size_t idx : test_indices) {
            std::printf("  %s%s%s\n", color::cyan(), tests[idx].name_str, color::reset());
        }
        
        std::printf("\n%s%zu test(s) would run", color::dim(), test_indices.size());
        if (cfg_.repeat > 1) {
            std::printf(" (%d repetitions = %zu total)",
                cfg_.repeat, test_indices.size() * cfg_.repeat);
        }
        std::printf("%s\n\n", color::reset());
        
        return 0;
    }
};

} // namespace scl::test

// =============================================================================
// Assertion Macros
// =============================================================================

// Helper to convert values to strings
namespace scl::test::detail {

template<typename T>
inline std::string to_string_impl(const T& value) {
    std::ostringstream oss;
    oss << std::boolalpha << value;
    return oss.str();
}

inline std::string to_string_impl(const char* value) {
    return value ? std::string("\"") + value + "\"" : "nullptr";
}

inline std::string to_string_impl(const std::string& value) {
    return "\"" + value + "\"";
}

inline std::string to_string_impl(std::nullptr_t) {
    return "nullptr";
}

template<typename T>
inline std::string to_string_impl(T* ptr) {
    if (!ptr) return "nullptr";
    std::ostringstream oss;
    oss << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr);
    return oss.str();
}

template<typename T>
inline std::string value_to_string(const T& value) {
    return to_string_impl(value);
}

} // namespace scl::test::detail

/// Basic assertion
#define SCL_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Assertion failed: " #expr); \
        } \
    } while (0)

/// Assertion with custom message
#define SCL_ASSERT_MSG(expr, msg) \
    do { \
        if (!(expr)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, msg); \
        } \
    } while (0)

/// Equality assertion
#define SCL_ASSERT_EQ(expected, actual) \
    do { \
        auto&& _exp = (expected); \
        auto&& _act = (actual); \
        if (!(_exp == _act)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected equality: " #expected " == " #actual, \
                ::scl::test::detail::value_to_string(_exp), \
                ::scl::test::detail::value_to_string(_act)); \
        } \
    } while (0)

/// Inequality assertion
#define SCL_ASSERT_NE(expected, actual) \
    do { \
        auto&& _exp = (expected); \
        auto&& _act = (actual); \
        if (_exp == _act) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected inequality: " #expected " != " #actual, \
                "not " + ::scl::test::detail::value_to_string(_exp), \
                ::scl::test::detail::value_to_string(_act)); \
        } \
    } while (0)

/// Less than assertion
#define SCL_ASSERT_LT(a, b) \
    do { \
        auto&& _a = (a); \
        auto&& _b = (b); \
        if (!(_a < _b)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected: " #a " < " #b, \
                "< " + ::scl::test::detail::value_to_string(_b), \
                ::scl::test::detail::value_to_string(_a)); \
        } \
    } while (0)

/// Less than or equal assertion
#define SCL_ASSERT_LE(a, b) \
    do { \
        auto&& _a = (a); \
        auto&& _b = (b); \
        if (!(_a <= _b)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected: " #a " <= " #b, \
                "<= " + ::scl::test::detail::value_to_string(_b), \
                ::scl::test::detail::value_to_string(_a)); \
        } \
    } while (0)

/// Greater than assertion
#define SCL_ASSERT_GT(a, b) \
    do { \
        auto&& _a = (a); \
        auto&& _b = (b); \
        if (!(_a > _b)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected: " #a " > " #b, \
                "> " + ::scl::test::detail::value_to_string(_b), \
                ::scl::test::detail::value_to_string(_a)); \
        } \
    } while (0)

/// Greater than or equal assertion
#define SCL_ASSERT_GE(a, b) \
    do { \
        auto&& _a = (a); \
        auto&& _b = (b); \
        if (!(_a >= _b)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected: " #a " >= " #b, \
                ">= " + ::scl::test::detail::value_to_string(_b), \
                ::scl::test::detail::value_to_string(_a)); \
        } \
    } while (0)

/// True assertion
#define SCL_ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected true: " #expr, "true", "false"); \
        } \
    } while (0)

/// False assertion
#define SCL_ASSERT_FALSE(expr) \
    do { \
        if (expr) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected false: " #expr, "false", "true"); \
        } \
    } while (0)

/// Null pointer assertion
#define SCL_ASSERT_NULL(ptr) \
    do { \
        auto&& _ptr = (ptr); \
        if (_ptr != nullptr) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected nullptr: " #ptr, "nullptr", \
                ::scl::test::detail::value_to_string(_ptr)); \
        } \
    } while (0)

/// Non-null pointer assertion
#define SCL_ASSERT_NOT_NULL(ptr) \
    do { \
        auto&& _ptr = (ptr); \
        if (_ptr == nullptr) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected non-null: " #ptr, "non-null", "nullptr"); \
        } \
    } while (0)

/// Floating-point near assertion
#define SCL_ASSERT_NEAR(expected, actual, tolerance) \
    do { \
        auto _exp = static_cast<double>(expected); \
        auto _act = static_cast<double>(actual); \
        auto _tol = static_cast<double>(tolerance); \
        if (std::abs(_exp - _act) > _tol) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected near: |" #expected " - " #actual "| <= " #tolerance, \
                std::to_string(_exp) + " ± " + std::to_string(_tol), \
                std::to_string(_act)); \
        } \
    } while (0)

/// String equality (case-sensitive)
#define SCL_ASSERT_STR_EQ(expected, actual) \
    do { \
        std::string _exp(expected); \
        std::string _act(actual); \
        if (_exp != _act) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "String mismatch", \
                "\"" + _exp + "\"", \
                "\"" + _act + "\""); \
        } \
    } while (0)

/// String contains
#define SCL_ASSERT_STR_CONTAINS(haystack, needle) \
    do { \
        std::string _hay(haystack); \
        std::string _ndl(needle); \
        if (_hay.find(_ndl) == std::string::npos) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "String does not contain substring", \
                "contains \"" + _ndl + "\"", \
                "\"" + _hay + "\""); \
        } \
    } while (0)

/// Exception assertion
#define SCL_ASSERT_THROWS(expr, exception_type) \
    do { \
        bool _caught = false; \
        try { \
            expr; \
        } catch (const exception_type&) { \
            _caught = true; \
        } catch (...) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Wrong exception type thrown by: " #expr, \
                #exception_type, "different exception"); \
        } \
        if (!_caught) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected exception not thrown: " #expr, \
                #exception_type, "no exception"); \
        } \
    } while (0)

/// No exception assertion
#define SCL_ASSERT_NO_THROW(expr) \
    do { \
        try { \
            expr; \
        } catch (const std::exception& e) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Unexpected exception: " #expr, \
                "no exception", e.what()); \
        } catch (...) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Unexpected exception: " #expr, \
                "no exception", "unknown exception"); \
        } \
    } while (0)

/// Fail immediately
#define SCL_FAIL(msg) \
    throw ::scl::test::TestException(__FILE__, __LINE__, msg)

/// Skip test
#define SCL_SKIP(reason) \
    throw ::scl::test::SkipException(reason)

/// Skip test if condition
#define SCL_SKIP_IF(condition, reason) \
    do { \
        if (condition) { \
            throw ::scl::test::SkipException(reason); \
        } \
    } while (0)

// =============================================================================
// Test Registration Macros
// =============================================================================

/// Begin test file
#define SCL_TEST_BEGIN \
    namespace { \
    static constexpr std::size_t _scl_test_base = __COUNTER__;

/// Define a test unit
#define SCL_TEST_UNIT(name) \
    static void _scl_test_##name(); \
    [[maybe_unused]] static bool _scl_reg_##name = []() { \
        constexpr std::size_t idx = __COUNTER__ - _scl_test_base - 1; \
        auto& test_info = ::scl::test::detail::get_tests()[idx]; \
        test_info.func = _scl_test_##name; \
        test_info.name_str = #name; \
        test_info.file = __FILE__; \
        test_info.line = __LINE__; \
        test_info.suite = ::scl::test::detail::current_suite(); \
        if (idx + 1 > ::scl::test::detail::get_count()) { \
            ::scl::test::detail::get_count() = idx + 1; \
        } \
        return true; \
    }(); \
    static void _scl_test_##name()

/// Define a test with tags
#define SCL_TEST_TAGGED(name, ...) \
    static void _scl_test_##name(); \
    [[maybe_unused]] static bool _scl_reg_##name = []() { \
        constexpr std::size_t idx = __COUNTER__ - _scl_test_base - 1; \
        auto& test_info = ::scl::test::detail::get_tests()[idx]; \
        test_info.func = _scl_test_##name; \
        test_info.name_str = #name; \
        test_info.file = __FILE__; \
        test_info.line = __LINE__; \
        test_info.suite = ::scl::test::detail::current_suite(); \
        test_info.tags = {__VA_ARGS__}; \
        if (idx + 1 > ::scl::test::detail::get_count()) { \
            ::scl::test::detail::get_count() = idx + 1; \
        } \
        return true; \
    }(); \
    static void _scl_test_##name()

/// Define a skipped test
#define SCL_TEST_SKIP(name, reason) \
    static void _scl_test_##name(); \
    [[maybe_unused]] static bool _scl_reg_##name = []() { \
        constexpr std::size_t idx = __COUNTER__ - _scl_test_base - 1; \
        auto& test_info = ::scl::test::detail::get_tests()[idx]; \
        test_info.func = _scl_test_##name; \
        test_info.name_str = #name; \
        test_info.file = __FILE__; \
        test_info.line = __LINE__; \
        test_info.suite = ::scl::test::detail::current_suite(); \
        test_info.skip = true; \
        test_info.skip_reason = reason; \
        if (idx + 1 > ::scl::test::detail::get_count()) { \
            ::scl::test::detail::get_count() = idx + 1; \
        } \
        return true; \
    }(); \
    static void _scl_test_##name()

/// Define an expected-failure test
#define SCL_TEST_XFAIL(name, reason) \
    static void _scl_test_##name(); \
    [[maybe_unused]] static bool _scl_reg_##name = []() { \
        constexpr std::size_t idx = __COUNTER__ - _scl_test_base - 1; \
        auto& test_info = ::scl::test::detail::get_tests()[idx]; \
        test_info.func = _scl_test_##name; \
        test_info.name_str = #name; \
        test_info.file = __FILE__; \
        test_info.line = __LINE__; \
        test_info.suite = ::scl::test::detail::current_suite(); \
        test_info.xfail = true; \
        test_info.xfail_reason = reason; \
        if (idx + 1 > ::scl::test::detail::get_count()) { \
            ::scl::test::detail::get_count() = idx + 1; \
        } \
        return true; \
    }(); \
    static void _scl_test_##name()

/// Define a test with custom timeout
#define SCL_TEST_TIMEOUT(name, timeout_ms_value) \
    static void _scl_test_##name(); \
    [[maybe_unused]] static bool _scl_reg_##name = []() { \
        constexpr std::size_t idx = __COUNTER__ - _scl_test_base - 1; \
        auto& test_info = ::scl::test::detail::get_tests()[idx]; \
        test_info.func = _scl_test_##name; \
        test_info.name_str = #name; \
        test_info.file = __FILE__; \
        test_info.line = __LINE__; \
        test_info.suite = ::scl::test::detail::current_suite(); \
        test_info.timeout_ms = timeout_ms_value; \
        if (idx + 1 > ::scl::test::detail::get_count()) { \
            ::scl::test::detail::get_count() = idx + 1; \
        } \
        return true; \
    }(); \
    static void _scl_test_##name()

/// Begin a test suite
#define SCL_TEST_SUITE(name) \
    namespace _scl_suite_##name { \
    [[maybe_unused]] static bool _scl_suite_init = []() { \
        ::scl::test::detail::current_suite() = #name; \
        return true; \
    }();

/// End a test suite
#define SCL_TEST_SUITE_END \
    [[maybe_unused]] static bool _scl_suite_cleanup = []() { \
        ::scl::test::detail::current_suite() = nullptr; \
        return true; \
    }(); \
    }

/// Test case within a suite (alias for SCL_TEST_UNIT)
#define SCL_TEST_CASE(name) SCL_TEST_UNIT(name)

/// End test file
#define SCL_TEST_END \
    } /* anonymous namespace */

/// Generate main() function with full CLI support
#define SCL_TEST_MAIN() \
    int main(int argc, char* argv[]) { \
        ::scl::test::parse_args(argc, argv); \
        ::scl::test::Runner runner; \
        return runner.run(); \
    }

// =============================================================================
// Fixture Support
// =============================================================================

/// Define a test fixture class
#define SCL_TEST_FIXTURE(fixture_name) \
    struct fixture_name

/// Test using a fixture
#define SCL_TEST_F(fixture_name, test_name) \
    static void _scl_test_##fixture_name##_##test_name(); \
    [[maybe_unused]] static bool _scl_reg_##fixture_name##_##test_name = []() { \
        constexpr std::size_t idx = __COUNTER__ - _scl_test_base - 1; \
        auto& test_info = ::scl::test::detail::get_tests()[idx]; \
        test_info.func = _scl_test_##fixture_name##_##test_name; \
        test_info.name_str = #fixture_name "." #test_name; \
        test_info.file = __FILE__; \
        test_info.line = __LINE__; \
        test_info.suite = #fixture_name; \
        if (idx + 1 > ::scl::test::detail::get_count()) { \
            ::scl::test::detail::get_count() = idx + 1; \
        } \
        return true; \
    }(); \
    static void _scl_test_##fixture_name##_##test_name##_impl(fixture_name& fixture); \
    static void _scl_test_##fixture_name##_##test_name() { \
        fixture_name fixture; \
        _scl_test_##fixture_name##_##test_name##_impl(fixture); \
    } \
    static void _scl_test_##fixture_name##_##test_name##_impl([[maybe_unused]] fixture_name& fixture)

// =============================================================================
// Parameterized Tests
// =============================================================================

/// Define parameterized test data
#define SCL_TEST_PARAMS(name, type, ...) \
    static const std::vector<type> _scl_params_##name = {__VA_ARGS__};

/// Parameterized test
#define SCL_TEST_P(name, params_name, param_var) \
    static void _scl_test_##name##_impl(const decltype(_scl_params_##params_name)::value_type& param_var); \
    [[maybe_unused]] static bool _scl_reg_##name = []() { \
        for (std::size_t i = 0; i < _scl_params_##params_name.size(); ++i) { \
            constexpr std::size_t base_idx = __COUNTER__ - _scl_test_base - 1; \
            std::size_t idx = base_idx + i; \
            if (idx >= ::scl::test::MAX_TEST_UNITS) break; \
            auto& test_info = ::scl::test::detail::get_tests()[idx]; \
            test_info.func = [i]() { \
                _scl_test_##name##_impl(_scl_params_##params_name[i]); \
            }; \
            static std::string names[256]; \
            names[i] = std::string(#name) + "[" + std::to_string(i) + "]"; \
            test_info.name_str = names[i].c_str(); \
            test_info.file = __FILE__; \
            test_info.line = __LINE__; \
            test_info.suite = ::scl::test::detail::current_suite(); \
            if (idx + 1 > ::scl::test::detail::get_count()) { \
                ::scl::test::detail::get_count() = idx + 1; \
            } \
        } \
        return true; \
    }(); \
    static void _scl_test_##name##_impl(const decltype(_scl_params_##params_name)::value_type& param_var)

// =============================================================================
// Benchmark Support
// =============================================================================

namespace scl::test {

struct BenchmarkResult {
    double min_ns = 0;
    double max_ns = 0;
    double avg_ns = 0;
    double median_ns = 0;
    double stddev_ns = 0;
    std::size_t iterations = 0;
};

template<typename Func>
inline BenchmarkResult benchmark(Func&& func, std::size_t iterations = 1000, 
                                  std::size_t warmup = 100) {
    // Warmup
    for (std::size_t i = 0; i < warmup; ++i) {
        func();
    }
    
    // Measure
    std::vector<double> times;
    times.reserve(iterations);
    
    for (std::size_t i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        func();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    
    // Calculate statistics
    std::sort(times.begin(), times.end());
    
    BenchmarkResult result;
    result.iterations = iterations;
    result.min_ns = times.front();
    result.max_ns = times.back();
    result.median_ns = times[iterations / 2];
    
    double sum = 0;
    for (double t : times) sum += t;
    result.avg_ns = sum / iterations;
    
    double variance = 0;
    for (double t : times) {
        double diff = t - result.avg_ns;
        variance += diff * diff;
    }
    result.stddev_ns = std::sqrt(variance / iterations);
    
    return result;
}

inline void print_benchmark(const char* name, const BenchmarkResult& result) {
    std::printf("  %s%-30s%s  ", color::cyan(), name, color::reset());
    std::printf("avg: %s%.2f ns%s  ", color::bold(), result.avg_ns, color::reset());
    std::printf("min: %.2f ns  ", result.min_ns);
    std::printf("max: %.2f ns  ", result.max_ns);
    std::printf("σ: %.2f ns\n", result.stddev_ns);
}

} // namespace scl::test

#define SCL_BENCHMARK(name, iterations) \
    do { \
        auto _bench_result = ::scl::test::benchmark([&]() { name; }, iterations); \
        ::scl::test::print_benchmark(#name, _bench_result); \
    } while (0)

#endif // SCL_TEST_HPP
