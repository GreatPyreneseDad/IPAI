# Contributing to IPAI

Thank you for your interest in contributing to IPAI! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences
- Show empathy towards other community members

## How to Contribute

### Reporting Issues

1. **Check existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - System information
   - Error messages and logs

### Suggesting Features

1. **Open a discussion** first for major features
2. **Provide use cases** and examples
3. **Consider implementation complexity**
4. **Align with project vision** of coherence preservation

### Code Contributions

#### Setting Up Development Environment

1. **Fork the repository**
```bash
git clone https://github.com/YOUR_USERNAME/IPAI.git
cd IPAI
git remote add upstream https://github.com/GreatPyreneseDad/IPAI.git
```

2. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Install dependencies**
```bash
# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Frontend
cd frontend
npm install
```

#### Development Guidelines

##### Python Code Style
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black formatter)
- Docstrings for all public functions

```python
def calculate_coherence(
    psi: float,
    rho: float,
    q: float,
    f: float
) -> CoherenceResult:
    """
    Calculate coherence using GCT formula.
    
    Args:
        psi: Internal consistency (0-1)
        rho: Accumulated wisdom (0-1)
        q: Moral activation energy (0-1)
        f: Social belonging (0-1)
        
    Returns:
        CoherenceResult with score and analysis
    """
```

##### TypeScript/React Style
- Use functional components
- Proper TypeScript types
- ESLint and Prettier compliance

```typescript
interface CoherenceProps {
  score: number
  level: CoherenceLevel
  onUpdate?: (score: number) => void
}

export const CoherenceIndicator: React.FC<CoherenceProps> = ({
  score,
  level,
  onUpdate
}) => {
  // Component implementation
}
```

#### Testing Requirements

##### Backend Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_coherence.py::test_gct_calculation
```

##### Frontend Tests
```bash
# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch
```

##### Test Coverage
- Maintain minimum 80% coverage
- Write tests for new features
- Update tests for bug fixes

#### Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add coherence visualization component
fix: resolve howlround detection edge case
docs: update API documentation
test: add integration tests for auth flow
refactor: simplify GCT calculation logic
style: format code with Black
chore: update dependencies
```

#### Pull Request Process

1. **Update documentation** for new features
2. **Add/update tests** as needed
3. **Ensure CI passes** all checks
4. **Keep PR focused** on a single concern
5. **Write clear PR description**:
   - What changes were made
   - Why changes were necessary
   - How to test changes
   - Related issues

##### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Documentation Contributions

#### Types of Documentation
- **API Documentation**: Endpoint descriptions
- **Code Documentation**: Inline comments and docstrings
- **User Guides**: How-to articles
- **Architecture Docs**: System design explanations

#### Documentation Style
- Clear and concise language
- Code examples where helpful
- Proper markdown formatting
- Spell check before submitting

### Translation Contributions

Help make IPAI accessible globally:

1. **Check existing translations** in `locales/`
2. **Use consistent terminology**
3. **Maintain context** for coherence concepts
4. **Test UI with translations**

## Development Workflow

### 1. Sync with Upstream
```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature
```

### 3. Make Changes
- Write code
- Add tests
- Update documentation

### 4. Run Quality Checks
```bash
# Backend
black src/
ruff check src/
mypy src/
pytest tests/

# Frontend
npm run lint
npm run typecheck
npm test
```

### 5. Commit Changes
```bash
git add .
git commit -m "feat: add amazing feature"
```

### 6. Push and Create PR
```bash
git push origin feature/your-feature
```

## Project Structure Guide

### Backend Structure
```
src/
├── api/v1/         # API endpoints
├── coherence/      # Core coherence logic
├── models/         # Data models
├── services/       # Business logic
└── utils/          # Helper functions
```

### Frontend Structure
```
src/
├── components/     # Reusable components
├── pages/          # Page components
├── services/       # API services
├── hooks/          # Custom hooks
└── contexts/       # React contexts
```

## Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

### Release Checklist
1. Update version in `package.json` and `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test Docker images
5. Tag release in git
6. Create GitHub release

## Community

### Getting Help
- **GitHub Issues**: Bug reports and features
- **Discussions**: General questions
- **Discord**: Real-time chat (coming soon)

### Code Reviews
- Be constructive and kind
- Focus on code, not person
- Suggest improvements
- Acknowledge good solutions

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- GitHub contributors page
- Release notes
- Project documentation

## Legal

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Open a new discussion

Thank you for helping make IPAI better!