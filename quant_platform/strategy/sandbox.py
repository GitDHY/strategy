"""
Safe execution sandbox for user-defined strategies.
Uses RestrictedPython to prevent malicious code execution.
"""

import sys
import threading
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager
import traceback
import concurrent.futures

# RestrictedPython imports
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Guards import safe_builtins as rp_safe_builtins
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem


def _safe_getattr(obj, name, default=None):
    """
    Safe getattr implementation for RestrictedPython.
    Blocks access to private/dunder attributes.
    """
    # Block access to private attributes (starting with _)
    if name.startswith('_'):
        raise AttributeError(
            f"Access to private attribute '{name}' is not allowed"
        )
    
    # Block access to potentially dangerous attributes
    dangerous_attrs = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__dict__', '__globals__', '__code__', '__closure__',
        '__self__', '__func__', 'func_globals', 'func_code',
        'gi_frame', 'gi_code', 'co_code', 'f_globals', 'f_locals',
    }
    if name in dangerous_attrs:
        raise AttributeError(
            f"Access to attribute '{name}' is not allowed"
        )
    
    return getattr(obj, name, default) if default is not None else getattr(obj, name)


def _safe_iter_unpack_sequence(it, spec, _getiter_):
    """
    Safe implementation of sequence unpacking for RestrictedPython.
    Used when code does: for a, b in items.items()
    
    spec can be:
    - An integer (simple case)
    - A dict like {'childs': (), 'min_len': 2} (RestrictedPython format)
    """
    # Extract the expected length from spec
    if isinstance(spec, dict):
        expected_len = spec.get('min_len', 2)
    elif isinstance(spec, int):
        expected_len = spec
    else:
        # Default to 2 for key-value pairs
        expected_len = 2
    
    for item in _getiter_(it):
        if isinstance(item, (list, tuple)):
            if len(item) != expected_len:
                raise ValueError(
                    f"not enough values to unpack (expected {expected_len}, got {len(item)})"
                )
            yield item
        else:
            # Try to convert to tuple for unpacking
            try:
                item_tuple = tuple(item)
                if len(item_tuple) != expected_len:
                    raise ValueError(
                        f"not enough values to unpack (expected {expected_len}, got {len(item_tuple)})"
                    )
                yield item_tuple
            except TypeError:
                raise TypeError(f"cannot unpack non-iterable {type(item).__name__} object")


class ExecutionTimeout(Exception):
    """Raised when strategy execution exceeds time limit."""
    pass


class SafetyViolation(Exception):
    """Raised when strategy code attempts unsafe operations."""
    pass


class StrategyError(Exception):
    """Raised when strategy code has errors."""
    pass


@contextmanager
def timeout_handler(seconds: int):
    """
    Context manager for execution timeout.
    Uses ThreadPoolExecutor for cross-platform and thread-safe timeout.
    This is a simple wrapper that just yields - actual timeout is handled in execute().
    """
    # Just yield - timeout is handled at a higher level using ThreadPoolExecutor
    yield


class SafeExecutor:
    """
    Sandbox executor for user strategy code.
    Provides restricted Python execution environment.
    """
    
    # Allowed built-in functions (safe subset)
    ALLOWED_BUILTINS = {
        # Basic types
        'True': True,
        'False': False,
        'None': None,
        
        # Type constructors
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        
        # Math functions
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'sorted': sorted,
        'reversed': reversed,
        
        # Type checking
        'isinstance': isinstance,
        'type': type,
        
        # Iteration helpers
        'all': all,
        'any': any,
        'filter': filter,
        'map': map,
        
        # String/formatting
        'format': format,
        'repr': repr,
        
        # Exceptions (for catching)
        'Exception': Exception,
        'ValueError': ValueError,
        'TypeError': TypeError,
        'KeyError': KeyError,
        'IndexError': IndexError,
    }
    
    # Explicitly blocked names
    BLOCKED_NAMES = {
        'open', 'file', 'input', 'raw_input',
        'exec', 'eval', 'compile', 'execfile',
        '__import__', 'importlib',
        'os', 'sys', 'subprocess', 'socket',
        'requests', 'urllib', 'http',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        '__builtins__', '__dict__', '__class__',
        '__bases__', '__subclasses__', '__mro__',
    }
    
    def __init__(self, timeout_seconds: int = 10):
        """
        Initialize safe executor.
        
        Args:
            timeout_seconds: Maximum execution time
        """
        self.timeout_seconds = timeout_seconds
    
    def _create_safe_globals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create restricted globals dictionary for execution.
        
        Args:
            context: Strategy context (API functions, data)
            
        Returns:
            Safe globals dictionary
        """
        safe_globals = {
            '__builtins__': self.ALLOWED_BUILTINS.copy(),
            '_getiter_': default_guarded_getiter,
            '_getitem_': default_guarded_getitem,
            '_getattr_': _safe_getattr,  # Allow safe attribute access
            '_iter_unpack_sequence_': _safe_iter_unpack_sequence,  # Allow tuple unpacking in loops
            '_write_': lambda x: x,  # Allow basic writes
            '_print_': self._safe_print,  # Safe print function
        }
        
        # Add context (strategy API)
        for key, value in context.items():
            if key not in self.BLOCKED_NAMES:
                safe_globals[key] = value
        
        return safe_globals
    
    def _safe_print(self, *args, **kwargs):
        """Safe print function that collects output."""
        # In sandbox, print just returns the string
        return ' '.join(str(a) for a in args)
    
    def compile_code(self, source: str) -> Any:
        """
        Compile strategy code in restricted mode.
        
        Args:
            source: Python source code
            
        Returns:
            Compiled code object
            
        Raises:
            StrategyError: If compilation fails
        """
        try:
            result = compile_restricted(
                source,
                filename='<strategy>',
                mode='exec'
            )
            
            # Handle different RestrictedPython versions
            # Newer versions return the code object directly or raise exceptions
            # Older versions return a result object with .code and .errors attributes
            if result is None:
                raise StrategyError("Compilation returned None - check code syntax")
            
            # Check if result has errors attribute (older RestrictedPython)
            if hasattr(result, 'errors') and result.errors:
                raise StrategyError(
                    f"Compilation errors:\n" + '\n'.join(result.errors)
                )
            
            # Get the actual code object
            if hasattr(result, 'code'):
                # Older RestrictedPython version
                if result.code is None:
                    # Check for errors in the result
                    errors = getattr(result, 'errors', []) or []
                    if errors:
                        raise StrategyError(f"Compilation errors:\n" + '\n'.join(errors))
                    raise StrategyError("Compilation failed - code object is None")
                return result.code
            else:
                # Newer version - result is the code object itself
                return result
            
        except SyntaxError as e:
            raise StrategyError(f"Syntax error at line {e.lineno}: {e.msg}")
        except StrategyError:
            raise
        except Exception as e:
            raise StrategyError(f"Compilation failed: {str(e)}")
    
    def execute(
        self,
        source: str,
        context: Dict[str, Any],
        entry_function: str = "strategy"
    ) -> Any:
        """
        Execute strategy code safely.
        
        Args:
            source: Python source code
            context: Strategy context (API, data)
            entry_function: Name of the main strategy function
            
        Returns:
            Result from strategy execution
            
        Raises:
            ExecutionTimeout: If execution exceeds time limit
            SafetyViolation: If unsafe operation attempted
            StrategyError: If strategy code has errors
        """
        # Compile code
        compiled = self.compile_code(source)
        
        # Create safe execution environment
        safe_globals = self._create_safe_globals(context)
        safe_locals = {}
        
        def _execute_code():
            """Inner function to execute in thread pool."""
            # Execute the module code
            exec(compiled, safe_globals, safe_locals)
            
            # Call the entry function if it exists
            if entry_function in safe_locals:
                func = safe_locals[entry_function]
                if callable(func):
                    return func()
            
            # If no entry function, look for 'result' variable
            if 'result' in safe_locals:
                return safe_locals['result']
            
            # Return the target_weights if set via context
            if 'ctx' in context and hasattr(context['ctx'], 'get_target_weights'):
                return context['ctx'].get_target_weights()
            
            return None
        
        try:
            # Execute with timeout using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_code)
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    return result
                except concurrent.futures.TimeoutError:
                    raise ExecutionTimeout(
                        f"Strategy execution exceeded {self.timeout_seconds} seconds"
                    )
                
        except ExecutionTimeout:
            raise
        except SafetyViolation:
            raise
        except StrategyError:
            raise
        except Exception as e:
            # Wrap execution errors
            tb = traceback.format_exc()
            raise StrategyError(f"Strategy execution error: {str(e)}\n{tb}")
    
    def validate_code(self, source: str) -> Dict[str, Any]:
        """
        Validate strategy code without executing.
        
        Args:
            source: Python source code
            
        Returns:
            Dictionary with validation results:
            - 'valid': bool
            - 'errors': list of error messages
            - 'warnings': list of warnings
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
        }
        
        # Check for blocked imports/names
        for blocked in self.BLOCKED_NAMES:
            if blocked in source:
                result['warnings'].append(
                    f"Code contains potentially unsafe name: '{blocked}'"
                )
        
        # Check for import statements
        if 'import ' in source:
            result['warnings'].append(
                "Import statements are not allowed in strategies. "
                "Use the provided context API instead."
            )
        
        # Try to compile
        try:
            self.compile_code(source)
        except StrategyError as e:
            result['valid'] = False
            result['errors'].append(str(e))
        
        return result
    
    def get_safe_api_docs(self) -> str:
        """
        Get documentation for the safe API available in strategies.
        
        Returns:
            API documentation string
        """
        return """
# Strategy API Reference

## Available Built-in Functions

### Types
- `int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, `set`

### Math
- `abs(x)` - Absolute value
- `round(x, n)` - Round to n decimal places
- `min(*args)`, `max(*args)` - Minimum/maximum
- `sum(iterable)` - Sum of values

### Iteration
- `len(x)` - Length
- `range(start, stop, step)` - Range iterator
- `enumerate(iterable)` - Enumerate with indices
- `zip(*iterables)` - Zip iterables together
- `sorted(iterable)` - Sort values
- `all(iterable)`, `any(iterable)` - Boolean tests

## Context API (ctx)

### Data Access
- `ctx.get_price(ticker, lookback)` - Get price series
- `ctx.get_prices(tickers, lookback)` - Get multiple price series
- `ctx.vix()` - Get VIX series

### Indicators
- `ctx.ma(ticker, period)` - Simple Moving Average
- `ctx.ema(ticker, period)` - Exponential Moving Average
- `ctx.rsi(ticker, period)` - RSI indicator
- `ctx.atr(ticker, period)` - Average True Range
- `ctx.volatility(ticker, period)` - Rolling volatility

### Portfolio
- `ctx.get_current_weights()` - Current portfolio weights
- `ctx.set_target_weights(weights)` - Set target allocation

### Properties
- `ctx.current_date` - Current simulation date
- `ctx.tickers` - Available tickers

## Blocked Operations
- File system access (open, file)
- Network access (requests, socket)
- System commands (os, subprocess)
- Dynamic code execution (exec, eval)
- Module imports (import)
"""
