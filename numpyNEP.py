import numpy as np
from numpy import *

__all__ = np.__all__

class OverrideableUfunc(object):
    def __init__(self, real_ufunc):
        self._real_ufunc = real_ufunc

    # This handles readable attributes like .nargs, .types, etc.:
    def __getattr__(self, attr):
        return getattr(self._real_ufunc, attr)

    def _forward(self, method, args, kwargs):
        if hasattr(kwargs.get("out"), "_ufunc_override_"):
            return kwargs["out"]._ufunc_override_(self, method, "out", args, kwargs)
        for i, arg in enumerate(args):
            if hasattr(arg, "_ufunc_override_"):
                return arg._ufunc_override_(self, method, i, args, kwargs)
        return getattr(self._real_ufunc, method)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._forward("__call__", args, kwargs)

    def outer(self, *args, **kwargs):
        return self._forward("outer", args, kwargs)

    def reduce(self, *args, **kwargs):
        return self._forward("reduce", args, kwargs)

    def reduceat(self, *args, **kwargs):
        return self._forward("reduceat", args, kwargs)

    def accumulate(self, *args, **kwargs):
        return self._forward("accumulate", args, kwargs)

# For every ufunc in np, create a wrapped version in our namespace
for name, obj in np.__dict__.iteritems():
    if isinstance(obj, ufunc):
        globals()[name] = OverrideableUfunc(obj)

class NAType(object):
    def __new__(cls):
        if not hasattr(cls, "_NA_object"):
            cls._NA_object = object.__new__(cls)
        return cls._NA_object

    def __repr__(self):
        return "NA"

    # XX we don't bother implementing the actual payload/dtype stuff. This is
    # enough to let the tests pass, and it's not clear that this stuff serves
    # any purpose anyway.
    def __call__(self, payload=None, dtype=None):
        return NA

    payload = None

    def __nonzero__(self):
        raise ValueError, "truth value of an NA is ambiguous"

NA = NAType()

# Many numpy global functions simply delegate to a ndarray method of the same
# name. Okay, we can override the methods. But:
#   -- We want to coerce things like lists into NEPArrays, not ndarrays,
#      before calling the method
#   -- numpy's versions of these functions swallow extra arguments, like
#      skipna
def _method_delegator(method):
    return (lambda a, *args, **kwargs:
                getattr(asanyarray(a), method)(*args, **kwargs))
for name in np.core.fromnumeric.__all__:
    globals()[name] = _method_delegator(name)


class NEPArrayFlags(object):
    def __init__(self, nep_array):
        self._nep_array = nep_array

    def __getitem__(self, key):
        if key == "MASKNA":
            return self._nep_array._valid is not None
        elif key == "OWNMASKNA":
            return self._nep_array._ownmaskna
        else:
            return np.ndarray.flags.__get__(self._nep_array)[key]

    def __setitem__(self, key, value):
        if key == "MASKNA":
            if value:
                self._nep_array._add_valid()
            else:
                # self._nep_array._valid = None
                # self._nep_array._ownmaskna = False
                raise ValueError, "naughty naughty, can't remove a mask"
        elif key == "OWNMASKNA":
            if not value:
                raise ValueError, "can't disown a mask"
            if self._nep_array._valid is not None:
                if not self._nep_array._ownmaskna:
                    self._nep_array._add_valid(self._nep_array._valid.copy())
            else:
                self._nep_array._add_valid()
        else:
            np.ndarray.flags.__get__(self._nep_array)[key] = value

    def __getattr__(self, key):
        return self[key.upper()]

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self[key.upper()] = value

def _with_valid(data, valid, own=True):
    array = np.asarray(data).view(type=NEPArray)
    if valid is not None:
        array._add_valid(valid, own=own)
    return array

def _broadcast_to(arr, target):
    target = np.asarray(target)
    arr = np.asarray(arr)
    broadcast = np.lib.stride_tricks.broadcast_arrays(*[arr, target])
    if broadcast[1].shape != target.shape:
        raise ValueError, "shape mismatch"
    return broadcast[0]

class NEPArray(np.ndarray):
    # We always instantiate with no mask.
    # _valid == None -> no mask allocated
    # _ownmaskna is just a semi-meaningless flag
    def __array_finalize__(self, obj):
        self._valid = None
        self._ownmaskna = False

    @property
    def _debug_data(self):
        return np.asarray(self)

    def copy(self, order="C", maskna=None):
        new = np.ndarray.copy(self, order=order)
        if self._valid is not None:
            if maskna in (True, None):
                return _with_valid(new, self._valid.copy())
            else:
                # remove mask
                if self._valid is not None and not np.all(self._valid):
                    raise ValueError, "can't remove mask from array with NAs!"
                return _with_valid(new, None)
        else:
            new = new.view(type=NEPArray)
            if maskna == True:
                new._add_valid()
            return new

    @property
    def flags(self):
        return NEPArrayFlags(self)

    # We don't bother with __array_wrap__ and __array_finalize__ -- instead we
    # override ufuncs and indexing directly.

    def _effective_valid(self):
        if self._valid is None:
            true = np.asarray([True])
            return np.lib.stride_tricks.as_strided(true,
                                                   self.shape,
                                                   (0,) * self.ndim)
        else:
            return self._valid

    def _add_valid(self, valid=None, own=True):
        if valid is None:
            valid = np.ones(self.shape, dtype=bool)
        valid = np.asarray(valid)
        assert self.shape == valid.shape
        assert valid.dtype == np.dtype(bool)
        data_steps = np.asarray(self.strides) // self.dtype.itemsize
        wanted_strides = tuple(valid.itemsize * data_steps)
        # mask has different ordering from data, they could get out-of-sync
        # when reshaping, so recreate the mask with the same ordering.
        if wanted_strides != valid.strides:
            print "rearranging mask: strides %r -> %r" % (valid.strides, wanted_strides)
            new_valid = np.empty(self.size, dtype=bool)
            new_valid = np.lib.stride_tricks.as_strided(new_valid,
                                                        shape=self.shape,
                                                        strides=wanted_strides)
            new_valid[...] = valid[...]
            own = True
        self._valid = valid
        self._ownmaskna = own
        return self

    def _ufunc_override_(self, uf, method, i, args, kwargs):
        skipna = kwargs.pop("skipna", False)
        out = kwargs.pop("out", None)
        if "dtype" in kwargs and kwargs["dtype"] is None:
            del kwargs["dtype"]
        if method == "reduceat":
            arr_args = args[0]
            other_args = args[1:]
        else:
            arr_args = args
            other_args = ()
        valids = [asarray(arr_arg)._effective_valid() for arr_arg in arr_args]
        all_valid = valids[0]
        for valid in valids[1:]:
            all_valid &= valid
        where = asarray(kwargs.pop("where", True))

        # Sanity check the where= argument
        if method == "outer":
            wheres = where
        else:
            wheres = (where,)
        for where_arg in wheres:
            if not np.issubdtype(where_arg.dtype, np.dtype(bool)):
                raise ValueError, "where= argument must be boolean"
            if not np.all(where_arg._effective_valid()):
                raise ValueError, "don't know what to do with NA in where= argument"
        
        def dtype_from(value):
            if isinstance(value, np.ndarray):
                return value.dtype
            if isinstance(value, np.dtype):
                return value
            if isinstance(value, type):
                return np.dtype(value)
            return type(value)

        if method == "__call__":
            arr_args = np.lib.stride_tricks.broadcast_arrays(*arr_args)
            where = _broadcast_to(where, arr_args[0])
            if skipna:
                # Can't use &= b/c broadcasting leaves us with funny strides:
                where = where & all_valid 
            out_data = uf(*[arr[where] for arr in arr_args], **kwargs)
            if out is None:
                out = empty(arr_args[0].shape, dtype=dtype_from(out_data))
            out[where] = out_data
            out[where] = all_valid[where]
            return out
        elif method == "reduce":
            assert len(arr_args) == 1
            arr = arr_args[0]
            where = _broadcast_to(where, arr)
            if skipna:
                where = where & all_valid
            axis = kwargs.pop("axis", 0)
            keepdims = kwargs.pop("keepdims", False)
            if axis is None:
                axis = 0
                arr = arr.flatten()
                all_valid = all_valid.flatten()
                where = where.flatten()
            if isinstance(axis, int):
                axis = (axis,)
            assert isinstance(axis, tuple)
            out_shape = tuple([size for (i, size) in enumerate(arr.shape)
                               if i not in axis])
            if keepdims:
                final_out_shape = list(arr.shape)
                for i in axis:
                    final_out_shape[i] = 1
                final_out_shape = tuple(final_out_shape)
            else:
                final_out_shape = out_shape
            if out is not None:
                final_out_shape = out.shape
                out.resize(out_shape)
            if out is None:
                dtype_check = uf.reduce(np.asarray(arr)[:2, ...])
                out = empty(out_shape, dtype=dtype_from(dtype_check),
                            maskna=(arr._valid is not None and not skipna))
            for i in xrange(np.prod(out_shape, dtype=int)):
                if out_shape:
                    out_idx = np.unravel_index(i, out_shape)
                    in_idx = list(out_idx)
                    for i in sorted(axis):
                        in_idx.insert(i, slice(0))
                    in_idx = tuple(in_idx)
                else:
                    out_idx = None
                    in_idx = None
                if not np.any(where[in_idx]):
                    continue
                if skipna:
                    valid = True
                else:
                    valid = np.logical_and.reduce(arr._effective_valid()[in_idx][where[in_idx]])
                if valid:
                    value = uf.reduce(np.asarray(arr)[in_idx][where[in_idx]],
                                      **kwargs)
                else:
                    value = NA
                if value is NA and out._valid is None:
                    raise ValueError, "can't store NA in out array with maskna=False"
                out[out_idx] = value
            out.resize(final_out_shape)
            return out
        elif method in ("accumulate", "outer", "reduceat"):
            # XX wouldn't be too hard to add these. They aren't supported by
            # Mark's code in master either.
            for arr in arr_args:
                if not np.all(arr._effective_valid()):
                    raise TypeError, "ufunc.%s not yet supported for masked arrays" % (method,)
            if not np.all(where):
                raise ValueError, "where= not yet supported for ufunc.%s" % (method,)
            if out is not None:
                kwargs["out"] = out
            base_arr_args = tuple([np.asarray(arr) for arr in arr_args])
            return asarray(getattr(uf, method)(*(base_arr_args + other_args),
                                               **kwargs))

    # For these in-place operations, we always mutate the array first, so
    # if there is an error the mask is left unchanged. OTOH if mutating
    # the underlying array works, then mutating the mask should always
    # work too, since they are always the same shape.
    def _set_shape(self, shape):
        np.ndarray.shape.__set__(self, shape)
        if self._valid is not None:
            self._valid.shape = shape
    shape = property(np.ndarray.shape.__get__, _set_shape)

    def _set_strides(self, strides):
        assert np.all(np.asarray(strides) % self.dtype.itemsize == 0)
        valid_strides = tuple(np.asarray(strides) // self.dtype.itemsize
                              * np.dtype(bool).itemsize)
        np.ndarray.strides.__set__(self, strides)
        if self._valid is not None:
            self._valid.strides = valid_strides
    strides = property(np.ndarray.strides.__get__, _set_strides)

    def resize(self, *args, **kwargs):
        np.ndarray.resize(self, *args, **kwargs)
        if self._valid is not None:
            self._valid = self._valid.reshape(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        new = self.view()
        new.resize(*args, **kwargs)
        return new

    def squeeze(self, *args, **kwargs):
        squeezed = np.asarray(self).squeeze(*args, **kwargs)
        squeezed_valid = None
        if self._valid is not None:
            squeezed_valid = self._valid.squeeze(*args, **kwargs)
        return _with_valid(squeezed, squeezed_valid, own=False)

    def swapaxes(self, axis1, axis2):
        swapped = np.asarray(self).swapaxes(axis1, axis2)
        swapped_valid = None
        if self._valid is not None:
            swapped_valid = self._valid.swapaxes(axis1, axis2)
        return _with_valid(swapped, swapped_valid, own=False)

    def transpose(self, *axes):
        transposed = np.asarray(self).transpose(*axes)
        transposed_valid = None
        if self._valid is not None:
            transposed_valid = self._valid.transpose(*axes)
        return _with_valid(transposed, transposed_valid, own=False)

    def ravel(self, order="C"):
        if self._valid is None:
            # This supports all the orderings
            return asarray(np.asarray(self).ravel(order=order))
        else:
            # XX
            assert order in ("C", "F")
            if ((order == "C" and self.flags.c_contiguous)
                or (order == "F" and self.flags.f_contiguous)):
                return _with_valid(np.ndarray.ravel(self).ravel(order=order),
                                   self._valid.ravel(), own=False)
            return self.flatten(order=order)

    @property
    def T(self):
        return self.transpose()

    def repeat(self, *args, **kwargs):
        new = np.ndarray.repeat(self, *args, **kwargs)
        valid = None
        if self._valid is not None:
            valid = self._valid.repeat(self, *args, **kwargs)
        return _with_valid(new, valid)

    def mean(self, *args, **kwargs):
        numerators = sum(self, *args, **kwargs)
        kwargs.pop("out", None)
        denominators = sum(self._effective_valid(), *args, **kwargs)
        numerators /= denominators
        return numerators

    def std(self, *args, **kwargs):
        return sqrt(self.var(*args, **kwargs))

    def var(self, axis=None, dtype=None, out=None, ddof=0, skipna=False,
            keepdims=False):
        raise NotImplementedError
        
    def __contains__(self, other):
        if other is NA:
            return np.any(~self._effective_valid())
        return np.ndarray.__contains__(self, other)

    def choose(self, choices, out=None, **kwargs):
        choices = [asarray(c) for c in choices]
        data = np.ndarray.choose(self, choices)
        valid = np.ndarray.choose(self, [c._effective_valid() for c in choices])
        return _with_valid(data, valid)

    # clip is fine as-is

    # XX not supported (by either this hack or Mark's code in master):
    #   argmax, argmin, argsort, sort, searchsorted
    # XX not supported by this hack, status in Mark's code unchecked:
    #   compress
    #   put, take
    #   tofile, tolist

    def view(self, dtype=None, type=None, maskna=None, ownmaskna=False):
        # This thing's calling conventions are totally annoying to emulate
        if isinstance(dtype, __builtins__["type"]) and issubclass(dtype, np.ndarray):
            type = dtype
            dtype = None
        if dtype is not None:
            dtype = np.dtype(dtype)
        kwargs = {}
        if type is not None:
            kwargs["type"] = type
        if dtype is not None:
            kwargs["dtype"] = dtype
        if (dtype is not None
            and dtype.itemsize != self.dtype.itemsize
            and self._valid is not None):
            raise TypeError, "can't view masked array using mismatched itemsize"
        arr = np.ndarray.view(self, **kwargs)
        assert isinstance(arr, NEPArray)
        if ownmaskna:
            maskna = True
        if self._valid is not None:
            if maskna in (None, True):
                if ownmaskna:
                    arr._add_valid(self._valid.copy())
                else:
                    arr._add_valid(self._valid, own=False)
            else:
                raise ValueError, "can't remove mask"
        else:
            if maskna == True:
                arr._add_valid()
        return arr

    def astype(self, dtype, *args, **kwargs):
        kwargs["subok"] = True
        # We don't want to call astype on masked out elements -- if they are
        # not convertible then that's okay.
        out = empty(self.shape, dtype=dtype, maskna=self.flags.maskna)
        valid = self._effective_valid()
        out[valid] = np.asarray(self)[valid].astype(dtype, *args, **kwargs)
        return out

    def __getitem__(self, key):
        data = np.ndarray.__getitem__(self, key)
        if ((isinstance(key, int) and self.ndim == 1)
            or (isinstance(key, tuple)
                and np.all([isinstance(key_part, int) for key_part in key]))):
            # Scalar indexing, so don't wrap this back up in an array
            if self._effective_valid()[key]:
                return data
            else:
                return NA
        valid = None
        own = False
        if self._valid is not None:
            valid = self._valid[key]
            # Figure out whether this created a view or a copy:
            new_final_base = valid
            while new_final_base.base is not None:
                new_final_base = new_final_base.base
            old_final_base = self._valid
            while old_final_base.base is not None:
                old_final_base = old_final_base.base
            if new_final_base is old_final_base:
                own = False
        return _with_valid(data, valid, own=own)

    def __setitem__(self, key, value):
        value = asarray(value)
        if self._valid is None:
            if np.any(~value._effective_valid()):
                raise ValueError, "cannot assign NA to array with maskna=False"
            np.ndarray.__setitem__(self, key, np.asarray(value))
        else:
            valid = _broadcast_to(value._effective_valid(), np.asarray(self)[key])
            if valid.size == 1:
                if valid:
                    np.asarray(self)[key] = np.asarray(value)
            else:
                np.asarray(self)[key][valid] = (
                    np.asarray(value)[value._effective_valid()])
            self._valid[key] = value._effective_valid()

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __setslice__(self, i, j, value):
        self.__setitem__(slice(i, j), value)

    def _objarray_with_NAs(self):
        object_array = np.asarray(self, dtype=object)
        if object_array.ndim > 0:
            object_array[~self._effective_valid()] = NA
        else:
            if not self._effective_valid().item():
                object_array = np.array(NA, dtype=object)
        return object_array

    def tolist(self):
        return self._objarray_with_NAs().tolist()

    def __repr__(self):
        extra = ""
        boring_dtypes = [np.dtype(float), np.dtype(int)]
        if (not np.any(self._effective_valid())
            or self.dtype not in boring_dtypes):
            extra = ", dtype=%s" % (self.dtype,)
        if self._valid is not None and np.all(self._valid):
            extra += ", maskna=True"
        return repr(self._objarray_with_NAs()).replace(", dtype=object",
                                                       extra)

    def diagonal(self):
        arr = np.ndarray.diagonal(self)
        if self._valid is not None:
            arr._add_valid(self._valid.diagonal())
        return arr

    def dot(self, *args, **kwargs):
        return dot(self, *args, **kwargs)

    def fill(self, value):
        np.ndarray.fill(self, value)
        if self._valid is not None:
            self._valid.fill(True)

    @property
    def flat(self):
        return self.flatten()

    def flatten(self, order="C"):
        assert self._valid is None or order != "A"
        data = np.ndarray.flatten(self, order=order)
        valid = None
        if self._valid is not None:
            valid = self._valid.flatten(order=order)
        return _with_valid(data, valid)

    @property
    def imag(self):
        return _with_valid(np.ndarray.imag.__get__(self), self._valid,
                           own=False)

    @property
    def real(self):
        return _with_valid(np.ndarray.real.__get__(self), self._valid,
                           own=False)

    def item(self, *args):
        if self._effective_valid().item(*args):
            return np.ndarray.item(self, *args)
        else:
            return NA

    def itemset(self, *args):
        if args[-1] is NA:
            self._valid.itemset(*(args[:-1] + (False,)))
        else:
            self._valid.itemset(*(args[:-1] + (True,)))
            np.ndarray.itemset(self, *args)

    # XX: what the heck should nonzero() do with an NA?

    # The rest is just noise needed to make sure ufunc-like methods dispatch
    # via OverrideableUfunc instead of going straight to the real numpy
    # ufuncs, and could be removed if ordinary ufuncs become overrideable:
    
    def __divmod__(self, other):
        return (divide(self, other), remainder(self, other))

    def ptp(self, *args, **kwargs):
        return self.max(*args, **kwargs) - self.min(*args, **kwargs)

    def __nonzero__(self):
        # XX Mark's code handles this case incorrectly, a 0-d array
        # containing NA is treated as True (but the NA scalar raises an error)
        if self.size == 1 and not self._effective_valid():
            raise ValueError, "truth value of an NA is ambiguous"
        # If our size is 1, then this checks the one value available (which is
        # valid, or else we'd have caught it up above); otherwise it just
        # throws an error. So no need for fancy handling:
        return np.ndarray.__nonzero__(self)

    # Arity agnostic:
    def _make_func_delegator(func):
        return lambda self, *args, **kwargs: func(self, *args, **kwargs)
    # These only make sense for binary methods/functions:
    def _make_reverse_func_delegator(func):
        return lambda self, other: func(other, self)
    def _make_inplace_func_delegator(func):
        return lambda self, other: func(self, other, out=self)

    for (method, func) in [("sum", add.reduce),
                           ("all", logical_and.reduce),
                           ("any", logical_or.reduce),
                           ("max", maximum.reduce),
                           ("min", minimum.reduce),
                           ("conj", conjugate),
                           ("conjugate", conjugate),
                           ("cumprod", multiply.accumulate),
                           ("cumsum", add.accumulate),
                           ("prod", multiply.reduce),
                           ]:
        locals()[method] = _make_func_delegator(func)

    for (special, func) in [("abs", abs),
                            ("add", add),
                            ("and", bitwise_and),
                            ("div", divide),
                            ("eq", equal),
                            ("floordiv", floor_divide),
                            ("ge", greater_equal),
                            ("gt", greater),
                            ("invert", bitwise_not),
                            ("le", less_equal),
                            ("lshift", left_shift),
                            ("lt", less),
                            ("mod", remainder),
                            ("mul", multiply),
                            ("ne", not_equal),
                            ("neg", negative),
                            ("or", bitwise_or),
                            ("pow", pow),
                            ("sub", subtract),
                            ("truediv", true_divide),
                            ("xor", bitwise_xor),
                            ]:
        locals()["__%s__" % (special,)] = _make_func_delegator(func)
        # There is no such thing as e.g. "__rabs__", but defining it doesn't
        # do any harm either:
        locals()["__r%s__" % (special,)] = _make_reverse_func_delegator(func)
        # Likewise for in-place operations:
        locals()["__i%s__" % (special,)] = _make_inplace_func_delegator(func)

    del _make_func_delegator
    del _make_reverse_func_delegator
    del _make_inplace_func_delegator

ndarray = NEPArray

def isavail(arr):
    return np.copy(asarray(arr)._effective_valid())
    
def isna(arr):
    return np.logical_not(asarray(arr)._effective_valid())

def dot(a, b, out=None):
    out = asarray(np.dot(a, b, out=out))
    if a._valid is not None or b._valid is not None:
        valid = np.logical_not(np.dot(np.logical_not(a._effective_valid()),
                                     np.logical_not(b._effective_valid())))
        out._add_valid(valid)
    return out

def array(array_like, dtype=None, copy=True, order=None, subok=False,
          ndmin=0, maskna=None, ownmaskna=False):
    a = np.array(array_like, dtype=dtype, copy=copy, order=order, subok=True,
                 ndmin=ndmin)
    # This handles cases like array([1, 2, NA]), where we need to both do
    # dtype detection and extract the NAs to create a mask.
    if (not isinstance(array_like, np.ndarray)
        and a.dtype == np.dtype(object)
        and NA in np.asarray(a)):
        if maskna == False:
            raise ValueError, "found NA with maskna=False"
        data = np.asarray(a)
        if data.ndim == 0:
            return _with_valid(np.array(0.0), np.array(False))
        valid = (data != NA)
        non_NAs = data[valid]
        if non_NAs.size == 0:
            # array([NA]) gives a float array, just like array([])
            non_NA = 0.0
        else:
            non_NA = non_NAs[0]
        data[data == NA] = non_NA
        a = np.array(data.tolist())
        return _with_valid(a, valid)
    # If we get here, we actually have either a NEPArray or something that was
    # converted into a plain old array without any NAs in it
    if not isinstance(a, NEPArray):
        a = a.view(type=NEPArray)
    # Okay, now we have a NEPArray; just have to implement the maskna flags.
    if copy and maskna:
        ownmaskna = True
    if ownmaskna:
        maskna = True
    if isinstance(array_like, NEPArray) and array_like._valid is not None:
        if copy and maskna in (True, None):
            maskna = True
            ownmaskna = True
        # There is a mask on the input that we need to recover.
        if array_like is a:
            # np.array passed the input through unchanged, so we need to check
            # it and possibly copy it
            if maskna and not a.flags.maskna:
                a = a.copy()
                a._add_valid()
            if ownmaskna and not a.flags.ownmaskna:
                a = a.copy()
                assert a.flags.ownmaskna
        else:
            # np.array did not pass through the input; therefore the mask was
            # lost. Recover it.
            if ownmaskna:
                a._add_valid(array_like._valid.copy())
            else:
                a._add_valid(array_like._valid, own=False)
    if maskna == True and a._valid is None:
        a._add_valid()
    if maskna == False and a._valid is not None:
        if np.all(a._valid):
            return _with_valid(np.asarray(a), None)
        else:
            raise ValueError, "found NA with maskna=False"
    assert not maskna or a.flags.maskna
    assert not ownmaskna or a.flags.ownmaskna
    return a

def asarray(*args, **kwargs):
    kwargs.setdefault("copy", False)
    return array(*args, **kwargs)

def asanyarray(*args, **kwargs):
    kwargs.setdefault("subok", True)
    return asarray(*args, **kwargs)

def _wrap_array_builder(func):
    def builder(*args, **kwargs):
        maskna = kwargs.pop("maskna", None)
        ownmaskna = kwargs.pop("ownmaskna", False)
        return asarray(func(*args, **kwargs), maskna=maskna,
                       ownmaskna=ownmaskna)
    return builder

for func_name in ["arange", "zeros", "ones", "zeros_like", "ones_like"]:
    globals()[func_name] = _wrap_array_builder(globals()[func_name])

def empty(*args, **kwargs):
    maskna = kwargs.pop("maskna", None)
    arr = asarray(np.empty(*args, **kwargs), maskna=maskna)
    if maskna:
        arr._valid.fill(False)
    return arr

def empty_like(*args, **kwargs):
    maskna = kwargs.pop("maskna", None)
    arr = asarray(np.empty_like(*args, **kwargs), maskna=maskna)
    if maskna:
        arr._valid.fill(False)
    return arr

max = maximum.reduce
min = minimum.reduce
def copy(*args, **kwargs):
    kwargs["copy"] = True
    return array(*args, **kwargs)
