# Contributing to SecureNet AI

Thank you for your interest in contributing to SecureNet AI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/SecureNet_AI.git
   cd SecureNet_AI
   ```
3. **Set up the development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running the Application
```bash
streamlit run Home.py
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
We use `black` for code formatting and `flake8` for linting:
```bash
black .
flake8 .
```

## Coding Standards

- **Python Version**: Python 3.10+
- **Code Style**: Follow PEP 8 guidelines
- **Documentation**: Add docstrings for all functions and classes
- **Type Hints**: Use type hints where appropriate
- **Comments**: Write clear, concise comments explaining complex logic

### Code Structure
```
- Keep functions focused and single-purpose
- Use meaningful variable and function names
- Maintain consistent naming conventions (snake_case for functions/variables)
- Keep files modular and organized
```

## Submitting Changes

1. **Ensure your code passes all tests**:
   ```bash
   pytest tests/
   ```

2. **Format your code**:
   ```bash
   black .
   flake8 .
   ```

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```
   
   Use clear, descriptive commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template with relevant information
   - Link any related issues

## Reporting Bugs

When reporting bugs, please include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs. **actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Error messages or logs** (if applicable)
- **Screenshots** (if applicable)

Use the Bug Report template when creating an issue.

## Feature Requests

We welcome feature requests! When submitting a feature request:

- **Describe the problem** you're trying to solve
- **Explain your proposed solution**
- **Provide use cases** and examples
- **Consider alternatives** you've thought about

Use the Feature Request template when creating an issue.

## Pull Request Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Update documentation**: Include relevant documentation updates
- **Add tests**: Ensure new features have appropriate test coverage
- **Follow the template**: Fill out the PR template completely
- **Be responsive**: Address review comments promptly

## Questions?

If you have questions or need help:
- Open an issue with the "question" label
- Check existing issues and discussions first
- Be respectful and patient

## Recognition

Contributors will be recognized in our project documentation and release notes.

Thank you for contributing to SecureNet AI! 🛡️
