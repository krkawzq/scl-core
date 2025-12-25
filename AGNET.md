# AGENT.md - AI Developer Guide for scl-core

This document serves as the authoritative guide for AI agents working on the `scl-core` project. It defines the architectural philosophy, coding standards, and collaborative protocols required to maintain the project's high-performance nature.

**Core Mission**: To build a high-performance, biological operator library with zero-overhead C++ kernels and a stable C-ABI surface for Python integration.

---

## 1. The Human-AI Collaboration Protocol

We strictly enforce a "Human-in-the-Loop" architecture. Code is not just logic; it is a collaborative artifact.

### 1.1 Documentation Standard
* **Style**: Doxygen Triple Slash (`///`).
* **Format**: Markdown embedded in comments.
* **Language**: English only.
* **Math**: LaTeX syntax (`$ formula $`) for mathematical descriptions.

---

## 2. C++ Kernel Architecture: "Zero-Overhead"

The internal C++ layer (`src/cpp`, `include/scl`) is designed for maximum throughput and minimal latency.

### 2.1 Memory Management Philosophy
* **No Hidden Allocations**: Kernel functions must operate on pre-allocated memory. Dynamic allocation (heap) inside a compute kernel is **strictly forbidden**.
* **Raw Pointers over STL**: Prefer C-style raw pointers (`T*`) and size parameters over `std::vector` or complex objects in hot paths.
    * *Reasoning*: Ensures ABI compatibility and eliminates container overhead.
* **Container Policy**: Use internal, stack-based, or zero-overhead custom structures. Standard Library containers (`std::map`, `std::string`) are permitted only in cold paths (config/initialization).

### 2.2 Parallelism: "Backend Agnostic"
* **Abstraction Layer**: Direct usage of `omp.h`, `tbb.h`, or `pthread` in kernels is **prohibited**.
* **Unified Interface**: All parallel loops must use the project's unified threading dispatch interface (`scl::threading::parallel_for`).
    * *Reasoning*: Decouples the mathematical logic from the execution backend, allowing the build system to switch between OpenMP, TBB, BS::thread_pool, or Serial modes without changing kernel code.

---

## 3. Binding Strategy: "The C-ABI Firewall"

We reject heavy binding generators (Pybind11/Nanobind) in favor of a manual, stable, and lightweight C-ABI.

### 3.1 The `extern "C"` Interface
* **Stability**: All exported functions must use `extern "C"` linkage to prevent C++ name mangling.
* **Return Type Protocol**: Functions must not return C++ objects or throw exceptions across the boundary.
    * **Success**: Return `nullptr`.
    * **Failure**: Return a pointer to an Error Instance (see Section 3.2).

### 3.2 Exception Containment
* **The Barrier**: Every C-ABI wrapper must contain a top-level `try-catch` block.
* **Error Registry**: C++ exceptions are caught and converted into a generic "Error Instance" (struct with code & message) managed by a global registry.
    * *Mechanism*: Python receives a pointer to this error instance, looks up the corresponding Python exception type in a shared table, and raises it.

---

## 4. Observability: "Pooled Telemetry"

Progress tracking and status reporting must be **asynchronous** and **non-intrusive**.

### 4.1 Decoupled Progress System
* **Concept**: Operators report progress to a side-channel system, independent of the main computation flow.
* **Mechanism**:
    * **Pooling**: Progress slots are managed by a global, pre-allocated pool to avoid allocation during execution.
    * **Macro Access**: Operators request a progress buffer via a dedicated macro (e.g., `SCL_GET_PROGRESS`).
* **Concurrency**:
    * Writing to progress counters must be **lock-free** (e.g., atomic increment or loose non-atomic increment).
    * Precision is secondary to performance. A slightly inaccurate progress bar is better than a stalled computation kernel.

---

## 5. Development Mindset

When generating code for `scl-core`, assume the role of a **Systems Engineer**:

1.  **Trust the Compiler**: Write clean, loop-friendly code that auto-vectorizes well.
2.  **Respect the Boundary**: Keep C++ pure; keep Python high-level. The C-ABI layer is the only bridge.
3.  **Defensive on Interfaces, Aggressive on Internals**: Be strict about pointer validity at the API entry point, but assume valid data inside the hot loops to avoid redundant checks.

---
