# New features implemented (above and beyond NEP-style masked NA support)
#   - general interface for overriding ufunc implementation
#   - where= (including for reduce)
#   - ufunc.reduce(..., axis=(i, j, ...))
#   - squeeze(axis=...)
#   - np.copyto
#   - ufunc.reduce works on scalars (e.g. logical_or.reduce(True))
# Missing functions:
#   - compress
#   - extract
#   - put
#   - einsum
#   - correlate
#   - probably others I missed, caveat emptor
# Known limitations:
#   - The numpy scalar types (e.g. np.float64) define their own infix
#     operators (e.g. __mul__), which may dispatch to ufuncs and can't be
#     overridden. Therefore np.float64(1) * np.array([1, np.NA]) uses the
#     built-in ufunc and drops the NA value. The only way to avoid this is to
#     make NEPArray stop inheriting from ndarray, and delegate instead. Maybe
#     this is a good idea in any case... (The other way to fix it would be to
#     implement the ufunc override logic in the core.)
#   - np.all() and np.any() propagate NAs, they don't have the special (NA |
#     True) == True logic.
#   - Haven't implemented payloads or dtypes for the NA object, it's just a
#     strict singleton like None.
#   - Haven't implemented casting= and preservena= arguments to copyto
#   - Haven't implemented the logic to make np.diagonal return a view. (This
#     has nothing to do with NAs, but it's tested by test_maskna.py).
# Known limitations that are shared with Mark's code:
#   - Not implemented: argmin, argmax, argsort, sort, searchsorted
#   - tofile and tostring ignore the mask
#   - np.logical_or, np.logical_and don't have any special handling for NAs
#     (Mark's code implements this only for np.any/np.all.)
# Known limitations of Mark's code that *aren't* limitations of this code:
#   - where= and masks work together ufunc.__call__
#   - ufunc.reduce supports where= (including with masks)
#   - tolist() returns NAs in appropriate places, instead of stripping off the
#     mask
#   - bool(np.array(np.NA)) is an error, just like bool(np.NA). (In Mark's
#     code, only the latter is an error.
# Other discrepancies between this and Mark's code:
#   - There's something funny going on with memory order and concatenate.
#     test_array_maskna_concatenate checks a memory order invariant that I'm
#     seeing fail even for plain old ndarray's.
#   - repr has different whitespace. Oh noes.
#   - I think the original test_array_maskna_setasflat is wrong! If the code
#     in master is passing it then I think that is bad.

if "_first_time" not in globals():
    _first_time = True

if _first_time:
    import numpy as np

def _basearray(a, dtype=None):
    # Unlike np.asarray, this does not normalize the order of arrays passed
    # through it.
    if (dtype is None
        or getattr(a, "dtype", None) == dtype):
        if type(a) is np.ndarray:
            return a
        if isinstance(a, np.ndarray):
            return np.ndarray.view(a, np.ndarray)
    return np.array(a, copy=False, dtype=dtype)

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
    if isinstance(obj, np.ufunc):
        globals()[name] = OverrideableUfunc(obj)

_special_methods_and_ufuncs = [
    ("abs", abs),
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
    ("pow", power),
    ("sub", subtract),
    ("truediv", true_divide),
    ("xor", bitwise_xor),
    ]

class NAType(object):
    def __new__(cls):
        if not hasattr(cls, "_NA_object"):
            cls._NA_object = object.__new__(cls)
        return cls._NA_object

    def __repr__(self):
        return "NA"

    # XX we don't bother implementing the actual payload/dtype stuff. This is
    # enough to let the tests pass, and it's not clear that it serves any
    # purpose anyway.
    def __call__(self, payload=None, dtype=None):
        return NA

    payload = None

    def _arithmetic_op(self, *args, **kwargs):
        return self

    def _make_arith(name):
        def _op(self, *args, **kwargs):
            print self, name
            return NA
        return _op

    for special, _ in _special_methods_and_ufuncs:
        locals()["__%s__" % (special,)] = _make_arith(special)
        locals()["__r%s__" % (special,)] = _make_arith(special)
        locals()["__i%s__" % (special,)] = lambda self, other: self

    def __nonzero__(self):
        raise ValueError, "truth value of an NA is ambiguous"

    def _ufunc_override_(self, *args, **kwargs):
        return asarray(self)._ufunc_override_(*args, **kwargs)

NA = NAType()

# Many numpy global functions simply delegate to a ndarray method of the same
# name. Okay, we can override the methods. But:
#   -- We want to coerce things like lists into NEPArrays, not ndarrays,
#      before calling the method
#   -- numpy's versions of these functions swallow extra arguments, like
#      skipna
# However, some of them have different default arguments than the
# corresponding method. (E.g. np.sum is the same as add.reduce, but with
# axis=None instead of axis=0 as default.) So we do default argument lookup
# based on the original function's signature.
def _get_real_callargs(func, args, kwargs):
    """Works out a normalized *args and **kwargs for calling the given
    function, using the default values it has set. Only required arguments are
    left in 'args'."""
    from inspect import getargspec, getcallargs
    arg_names, varargs_name, keywords_name, defaults = getargspec(func)
    if defaults is None:
        defaults = ()
    call_args = getcallargs(func, *args, **kwargs)
    required_arg_names = arg_names[:len(arg_names) - len(defaults)]
    # I don't know why getcallargs() has such a funky return value.
    required_args = []
    for required_arg_name in required_arg_names:
        required_args.append(call_args.pop(required_arg_name))
    required_args = tuple(required_args)
    if varargs_name in call_args:
        required_args += call_args.pop(varargs_name)
    # call_args now contains only the unrequired arguments, plus perhaps a
    # dict containing any extra unexpected, named, arguments. Merge those into
    # the same dict as the rest of the arguments.
    if keywords_name in call_args:
        call_args.update(call_args.pop(keywords_name))
    return required_args, call_args
def _method_delegator(original, method):
    def delegate(*args, **kwargs):
        new_kwargs = {}
        new_kwarg_names = ["skipna", "keepdims"]
        if method == "squeeze":
            new_kwarg_names.append("axis")
        for new_kwarg_name in new_kwarg_names:
            if new_kwarg_name in kwargs:
                new_kwargs[new_kwarg_name] = kwargs.pop(new_kwarg_name)
        args, kwargs = _get_real_callargs(original, args, kwargs)
        kwargs.update(new_kwargs)
        return getattr(asanyarray(args[0]), method)(*args[1:], **kwargs)
    return delegate
for name in np.core.fromnumeric.__all__:
    globals()[name] = _method_delegator(getattr(np.core.fromnumeric, name),
                                        name)


def _normalize_axes(ndim, axis):
    assert isinstance(axis, tuple)
    axis = list(axis)
    for i, ax in enumerate(axis):
        if ax < 0:
            axis[i] += ndim
    return tuple(axis)

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
    array = _basearray(data).view(type=NEPArray)
    if valid is not None:
        array._add_valid(valid, own=own)
    return array

def _broadcast_to(arr, target):
    target = _basearray(target)
    arr = _basearray(arr)
    broadcast = np.broadcast_arrays(*[arr, target])
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
        return _basearray(self)

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
            true = _basearray([True])
            return np.lib.stride_tricks.as_strided(true,
                                                   self.shape,
                                                   (0,) * self.ndim)
        else:
            return self._valid

    def _add_valid(self, valid=None, own=True):
        if valid is None:
            valid = np.ones(self.shape, dtype=bool)
        valid = _basearray(valid)
        assert self.shape == valid.shape
        assert valid.dtype == np.dtype(bool)
        data_steps = _basearray(self.strides) // self.dtype.itemsize
        wanted_strides = tuple(valid.itemsize * data_steps)
        # mask has different ordering from data, they could get out-of-sync
        # when reshaping, so recreate the mask with the same ordering.
        if wanted_strides != valid.strides:
            print "rearranging mask: strides %r -> %r" % (valid.strides, wanted_strides)
            new_valid = _basearray(np.empty(self.size, dtype=bool))
            new_valid = np.lib.stride_tricks.as_strided(new_valid,
                                                        shape=self.shape,
                                                        strides=wanted_strides)
            new_valid[...] = valid[...]
            valid = new_valid
            own = True
        self._valid = valid
        self._ownmaskna = own
        return self

    def _ufunc_override_(self, uf, method, i, args, kwargs):
        skipna = kwargs.pop("skipna", False)
        out = kwargs.pop("out", None)
        out_given = (out is not None)
        if "dtype" in kwargs and kwargs["dtype"] is None:
            del kwargs["dtype"]
        n_arr_args = {
            "reduce": 1,
            "reduceat": 1,
            "accumulate": 1,
            "outer": 2,
            "__call__": uf.nin,
            }
        arr_args = args[:n_arr_args[method]]
        arr_args = [asarray(arr_arg) for arr_arg in arr_args]
        other_args = args[n_arr_args[method]:]
        valids = np.broadcast_arrays(*[asarray(arr_arg)._effective_valid()
                                       for arr_arg in arr_args])
        all_valid = valids[0]
        for valid in valids[1:]:
            all_valid = all_valid & valid
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
            maskna = (not skipna
                      and np.any([arr_arg.flags.maskna for arr_arg in arr_args]))
            arr_args = np.broadcast_arrays(*arr_args)
            where = _broadcast_to(where, arr_args[0])
            # Can't use &= b/c broadcasting leaves us with funny strides:
            where_and_valid = _basearray(where & all_valid)
            use_args = tuple([arr[where_and_valid] for arr in arr_args]) + other_args
            out_data = uf(*use_args, **kwargs)
            if out is None:
                out = empty(arr_args[0].shape, dtype=dtype_from(out_data),
                            maskna=maskna)
            if out.ndim == 0:
                if where:
                    if all_valid:
                        out.itemset(out_data)
                    else:
                        if out._valid is None:
                            raise ValueError, "can't write NA to out= array"
                        out._valid.itemset(False)
            else:
                _basearray(out)[where_and_valid] = out_data
                if out._valid is not None:
                    if skipna:
                        out._valid[where_and_valid] = all_valid[where_and_valid]
                    else:
                        out._valid[where] = all_valid[where]
                else:
                    if not np.all(all_valid[where]):
                        raise ValueError, "can't write NA to out= array"
            if not out_given and out.ndim == 0:
                return out[()]
            else:
                return out
        elif method == "reduce":
            assert len(arr_args) == 1
            arr = arr_args[0]
            for i, argname in enumerate(["axis", "dtype", "out"]):
                if len(other_args) > i:
                    kwargs[argname] = other_args[i]
            where = _broadcast_to(where, arr)
            if skipna:
                where = where & all_valid
            axis = kwargs.pop("axis", 0)
            keepdims = kwargs.pop("keepdims", False)
            if axis is None:
                axis = tuple(range(arr.ndim))
            if isinstance(axis, int):
                axis = (axis,)
            axis = _normalize_axes(arr.ndim, axis)
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
            # reductions always have the same output and input dtypes
            dtype = kwargs.get("dtype", arr.dtype)
            if out is None:
                out = empty(out_shape, dtype=dtype,
                            maskna=(arr._valid is not None and not skipna))
            for i in xrange(np.prod(out_shape, dtype=int)):
                if out_shape:
                    out_idx = np.unravel_index(i, out_shape)
                    in_idx = list(out_idx)
                    for i in sorted(axis):
                        in_idx.insert(i, slice(None))
                    in_idx = tuple(in_idx)
                else:
                    out_idx = None
                    in_idx = None
                if not np.any(where[in_idx]):
                    if uf.identity is None:
                        raise ValueError, "zero-size reduction with no identity"
                    value = uf.identity
                else:
                    if skipna:
                        valid = True
                    else:
                        valid = np.logical_and.reduce(arr._effective_valid()[in_idx][where[in_idx]])
                    if valid:
                        reduce_data = _basearray(arr)[in_idx]
                        if reduce_data.ndim == 0:
                            # We know that 'where' is True from the check above
                            value = reduce_data
                        else:
                            value = uf.reduce(reduce_data[where[in_idx]],
                                              **kwargs)
                    else:
                        value = NA
                if value is NA and out._valid is None:
                    raise ValueError, "can't store NA in out array with maskna=False"
                out[out_idx] = value
            out.resize(final_out_shape)
            if not out_given and out.ndim == 0:
                return out[()]
            else:
                return out
        elif method in ("accumulate", "outer", "reduceat"):
            # XX wouldn't be too hard to add these. The aren't supported by
            # Mark's code in master either.
            for arr in arr_args:
                if not np.all(arr._effective_valid()):
                    raise TypeError, "ufunc.%s not yet supported for masked arrays" % (method,)
            if not np.all(where):
                raise ValueError, "where= not yet supported for ufunc.%s" % (method,)
            if out is not None:
                kwargs["out"] = out
            base_arr_args = tuple([_basearray(arr) for arr in arr_args])
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
        assert np.all(_basearray(strides) % self.dtype.itemsize == 0)
        valid_strides = tuple(_basearray(strides) // self.dtype.itemsize
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
        # This may or may not make a copy, depending on order=
        reshaped = np.ndarray.reshape(self, *args, **kwargs)
        valid = None
        if self._valid is not None:
            valid = self._valid.reshape(*args, **kwargs)
        return _with_valid(reshaped, valid, own=False)

    def squeeze(self, axis=None):
        if axis is None:
            axis = tuple([i for (i, l) in enumerate(self.shape) if l == 1])
        if not isinstance(axis, tuple):
            axis = (axis,)
        axis = _normalize_axes(self.ndim, axis)
        for i in axis:
            if self.shape[i] != 1:
                raise ValueError, "can't squeeze non-unitary dimension"
        mask = np.ones(self.ndim, dtype=bool)
        mask[_basearray(axis)] = False
        new = self.view()
        new.shape = tuple(_basearray(self.shape)[mask])
        return new

    def swapaxes(self, axis1, axis2):
        swapped = np.ndarray.swapaxes(self, axis1, axis2)
        swapped_valid = None
        if self._valid is not None:
            swapped_valid = self._valid.swapaxes(axis1, axis2)
        return _with_valid(swapped, swapped_valid, own=False)

    def transpose(self, *axes):
        transposed = _basearray(self).transpose(*axes)
        transposed_valid = None
        if self._valid is not None:
            transposed_valid = self._valid.transpose(*axes)
        return _with_valid(transposed, transposed_valid, own=False)

    def ravel(self, order="C"):
        if self._valid is None:
            # This supports all the orderings
            return asarray(_basearray(self).ravel(order=order))
        else:
            if ((order in ("C", "A") and self.flags.c_contiguous)
                or (order in ("F", "A") and self.flags.f_contiguous)
                or order == "K"):
                own = False
            else:
                own = True
            return _with_valid(np.ndarray.ravel(self, order=order),
                               self._valid.ravel(order=order), own=own)

    @property
    def T(self):
        return self.transpose()

    def repeat(self, *args, **kwargs):
        new = np.ndarray.repeat(self, *args, **kwargs)
        valid = None
        if self._valid is not None:
            valid = self._valid.repeat(self, *args, **kwargs)
        return _with_valid(new, valid)

    def mean(self, axis=None, dtype=None, out=None, skipna=False, keepdims=False):
        if dtype is None and np.issubdtype(self.dtype, np.integer):
            dtype = float
        numerators = asarray(sum(self, axis=axis, dtype=dtype, skipna=skipna,
                                 keepdims=keepdims, out=out))
        denominators = count_reduce_items(self, axis=axis,
                                          skipna=skipna, keepdims=keepdims)
        numerators /= denominators
        if numerators.ndim == 0 and out is None:
            return numerators[()]
        else:
            return numerators

    def std(self, *args, **kwargs):
        return sqrt(self.var(*args, **kwargs))

    def var(self, axis=None, dtype=None, out=None, ddof=0, skipna=False,
            keepdims=False):
        N = count_reduce_items(self, axis=axis, skipna=skipna, keepdims=keepdims)
        # asarray() is to work around a bug -- otherwise, N may have type
        # np.float64, which defines its own __mul__ operator, which means that
        # 'scale * SX' will call the true np.multiply instead of our
        # overridden version.
        scale = asarray(1. / (N - ddof))
        SX = sum(self, axis=axis, skipna=skipna, keepdims=keepdims)
        SX2 = sum(self ** 2, axis=axis, skipna=skipna, keepdims=keepdims)
        result = scale * SX2 - (scale * SX) ** 2
        if out:
            out[...] = result
            return out
        else:
            return result
        
    def __contains__(self, other):
        if other is NA:
            return np.any(~self._effective_valid())
        return np.ndarray.__contains__(self, other)

    def choose(self, choices, out=None, **kwargs):
        choices = [asarray(c) for c in choices]
        data = np.ndarray.choose(self, choices)
        valid = np.ndarray.choose(self, [c._effective_valid() for c in choices])
        return _with_valid(data, valid)

    def clip(self, a_min, a_max, out=None):
        if out is None:
            out = empty(self.shape, dtype=self.dtype, maskna=self.flags.maskna)
        for i in xrange(np.prod(self.shape)):
            if self.shape == ():
                idx = ()
            else:
                idx = np.unravel_index(i, self.shape)
            if not self._effective_valid()[idx]:
                out[idx] = NA
            else:
                if a_min is not None and self[idx] < a_min:
                    out[idx] = a_min
                elif a_max is not None and self[idx] > a_max:
                    out[idx] = a_max
                else:
                    out[idx] = self[idx]
        return out

    # XX not supported (by either this hack or Mark's code in master):
    #   argmax, argmin, argsort, sort, searchsorted
    # XX not supported by this hack, status in Mark's code unchecked:
    #   compress
    #   put

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
        #assert isinstance(arr, NEPArray)
        if ownmaskna:
            maskna = True
        if self._valid is not None:
            if maskna in (None, True):
                if ownmaskna:
                    arr._add_valid(self._valid.copy())
                else:
                    arr._add_valid(self._valid.view(), own=False)
            else:
                raise ValueError, "can't remove mask"
        else:
            if maskna == True:
                arr._add_valid()
        return arr

    def astype(self, dtype, *args, **kwargs):
        # We don't want to call astype on masked out elements -- if they are
        # not convertible then that's okay.
        out = empty(self.shape, dtype=dtype, maskna=self.flags.maskna)
        valid = self._effective_valid()
        out[valid] = _basearray(self)[valid].astype(dtype, *args, **kwargs)
        return out

    def _check_index(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        for piece in key:
            if isinstance(piece, (int, long, slice,
                                  type(Ellipsis), type(None))):
                # simple indexing, no problem
                continue
            else:
                piece = asarray(piece)
                # We should perhaps allow integer indexing by NA (returning an
                # NA). But we don't yet.
                if not np.all(piece._effective_valid()):
                    raise ValueError, "Can't index by NA"

    def __getitem__(self, key):
        self._check_index(key)
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

    def take(self, indices, axis=None, out=None, mode="raise"):
        data = np.ndarray.take(self, indices, axis=axis, mode=mode)
        if self._valid is not None:
            valid = np.ndarray.take(self._effective_valid(), indices,
                                    axis=axis, mode=mode)
        result = _with_valid(data, valid)
        if out is None:
            return result
        else:
            out[...] = result
            return out

    def __setitem__(self, key, value):
        self._check_index(key)
        value = asarray(value)
        if self._valid is None:
            if np.any(~value._effective_valid()):
                raise ValueError, "cannot assign NA to array with maskna=False"
            np.ndarray.__setitem__(self, key, _basearray(value))
        else:
            valid = _broadcast_to(value._effective_valid(),
                                  _basearray(self)[key])
            if valid.size == 1:
                if value._effective_valid():
                    _basearray(self)[key] = _basearray(value)
            else:
                if np.any(value._effective_valid()):
                    bool_valid_key = np.zeros(self.shape, dtype=bool)
                    bool_valid_key[key] = valid
                    _basearray(self)[bool_valid_key] = (
                        _basearray(value)[value._effective_valid()])
            self._valid[key] = value._effective_valid()

    def setasflat(self, arr):
        arr = asarray(arr).ravel(order="K")
        self.ravel(order="K")[:] = arr[:]

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __setslice__(self, i, j, value):
        self.__setitem__(slice(i, j), value)

    def _objarray_with_NAs(self):
        object_array = _basearray(self, dtype=object)
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

    __str__ = __repr__

    def diagonal(self, *args, **kwargs):
        arr = np.ndarray.diagonal(self, *args, **kwargs)
        if self._valid is not None:
            arr._add_valid(self._valid.diagonal(*args, **kwargs))
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
            if self._valid is None:
                raise ValueError, "can't assign NA to array with namask=False"
            self._valid.itemset(*(args[:-1] + (False,)))
        else:
            if self._valid is not None:
                self._valid.itemset(*(args[:-1] + (True,)))
            np.ndarray.itemset(self, *args)

    def nonzero(self):
        if not np.all(self._effective_valid()):
            raise ValueError, "nonzero() undefined on array with NAs"
        return np.ndarray.nonzero(self)

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

    for (special, func) in _special_methods_and_ufuncs:
        locals()["__%s__" % (special,)] = _make_func_delegator(func)
        # Not all of these are meaningful (e.g. __rabs__) but defining them
        # won't do any harm either:
        locals()["__r%s__" % (special,)] = _make_reverse_func_delegator(func)
        locals()["__i%s__" % (special,)] = _make_inplace_func_delegator(func)

    conjugate = conj = _make_func_delegator(conjugate)

    del _make_func_delegator
    del _make_reverse_func_delegator
    del _make_inplace_func_delegator

    def _make_reduction_method_replacement(replacement, takes_dtype):
        if takes_dtype:
            def wrapper_method(self, axis=None, dtype=None, out=None,
                               skipna=False, keepdims=False):
                return replacement(self, axis=axis, dtype=dtype, out=out,
                                   skipna=skipna, keepdims=keepdims)
        else:
            def wrapper_method(self, axis=None, out=None, skipna=False,
                               keepdims=False):
                return replacement(self, axis=axis, out=out, skipna=skipna,
                                   keepdims=keepdims)
        return wrapper_method
    for (method, func, takes_dtype) in [("sum", add.reduce, True),
                                        ("all", logical_and.reduce, False),
                                        ("any", logical_or.reduce, False),
                                        ("max", maximum.reduce, False),
                                        ("min", minimum.reduce, False),
                                        ("amax", maximum.reduce, False),
                                        ("amin", minimum.reduce, False),
                                        ("cumprod", multiply.accumulate, True),
                                        ("cumsum", add.accumulate, True),
                                        ("prod", multiply.reduce, True),
                                        ]:
        locals()[method] = _make_reduction_method_replacement(func, takes_dtype)

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
    # First, convert without a dtype= argument, to let the auto-detection work
    a = np.array(array_like, copy=copy, order=order, subok=True,
                 ndmin=ndmin)
    # This handles cases like array([1, 2, NA]), where we need to both do
    # dtype detection and extract the NAs to create a mask.
    NA_mask = np.zeros(a.shape, dtype=bool)
    a_ravel = a.ravel()
    a_ids = _basearray(map(id, a_ravel.tolist()))
    NA_mask[...] = (a_ids == id(NA)).reshape(NA_mask.shape)
    if (not isinstance(array_like, np.ndarray)
        and a.dtype == np.dtype(object)
        and np.any(NA_mask)):
        if maskna == False:
            raise ValueError, "found NA with maskna=False"
        if a.ndim == 0:
            return _with_valid(np.array(0.0), np.array(False))
        valid = ~NA_mask
        non_NAs = a[valid]
        if non_NAs.size == 0:
            # array([NA]) gives a float array, just like array([])
            non_NA = 0.0
        else:
            non_NA = non_NAs[0]
        a[NA_mask] = non_NA
        a = np.array(a.tolist(), dtype=dtype, order=order, ndmin=ndmin)
        return _with_valid(a, valid)
    # If we get here, we actually have either a NEPArray or something that was
    # converted into a plain old array without any NAs in it. But, we might
    # have been requested to change the dtype and not done so.
    if dtype is not None and dtype != a.dtype:
        a = np.array(array_like, dtype=dtype, copy=copy, order=order, subok=True,
                     ndmin=ndmin)
    if (not isinstance(a, NEPArray)
        or (not subok and type(a) != NEPArray)):
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
                a = a.view()
                a._add_valid()
            if ownmaskna and not a.flags.ownmaskna:
                a = _with_valid(_basearray(a), a._effective_valid().copy())
                assert a.flags.ownmaskna
        else:
            # np.array did not pass through the input; therefore the mask was
            # lost. Recover it. (.reshape() in case ndmin= was used to change
            # the array's shape.)
            if ownmaskna:
                a._add_valid(array_like._valid.copy().reshape(a.shape))
            else:
                a._add_valid(array_like._valid.reshape(a.shape), own=False)
    if maskna == True and a._valid is None:
        a._add_valid()
    if maskna == False and a._valid is not None:
        if np.all(a._valid):
            return _with_valid(_basearray(a), None)
        else:
            raise ValueError, "found NA with maskna=False"
    assert not maskna or a.flags.maskna
    assert not ownmaskna or a.flags.ownmaskna
    return a

def _wrap_array_builder(func):
    def builder(*args, **kwargs):
        maskna = kwargs.pop("maskna", None)
        ownmaskna = kwargs.pop("ownmaskna", False)
        return asarray(func(*args, **kwargs), maskna=maskna,
                       ownmaskna=ownmaskna)
    return builder

# This is only re-defined so that it will accept and pass on the new arguments
# like maskna.
def asarray(*args, **kwargs):
    kwargs.setdefault("copy", False)
    return array(*args, **kwargs)

# This is only re-defined so that it will accept and pass on the new arguments
# like maskna.
def asanyarray(*args, **kwargs):
    kwargs.setdefault("subok", True)
    return asarray(*args, **kwargs)

for func_name in ["arange", "zeros", "ones", "zeros_like", "ones_like",
                  "linspace", "logspace", "eye", "identity"]:
    globals()[func_name] = _wrap_array_builder(getattr(np, func_name))

def empty(*args, **kwargs):
    maskna = kwargs.pop("maskna", None)
    arr = np.empty(*args, **kwargs).view(NEPArray)
    if maskna:
        arr._add_valid()
        arr._valid.fill(False)
    return arr

def empty_like(*args, **kwargs):
    maskna = kwargs.pop("maskna", None)
    arr = asarray(np.empty_like(*args, **kwargs), maskna=maskna)
    if maskna:
        arr._valid.fill(False)
    return arr

max = amax
min = amin

def copy(*args, **kwargs):
    kwargs["copy"] = True
    return array(*args, **kwargs)

def concatenate(array_likes, axis=0):
    print array_likes, axis
    arrays = [asarray(array_like) for array_like in array_likes]
    data = np.concatenate(arrays, axis=axis)
    print "data.strides =", data.strides
    valid = None
    if np.any([arr._valid is not None for arr in arrays]):
        valid = np.concatenate([arr._effective_valid() for arr in arrays],
                               axis=axis)
    return _with_valid(data, valid)

def count_nonzero(a, *args, **kwargs):
    return sum(a != 0, dtype=int, *args, **kwargs)

def count_reduce_items(arr, axis=None, skipna=False, keepdims=False):
    # XX this has confusing semantics. AFAICT, they are:
    # - we unconditionally return a scalar, unless:
    #   - skipna=True, and
    #   - the given array has a mask
    if skipna and arr._valid is not None:
        count_arr = _basearray(arr._effective_valid(), dtype=int)
    else:
        count_arr = np.ones(arr.shape, dtype=int)
    result = sum(count_arr, axis=axis, keepdims=keepdims)
    if not skipna or arr._valid is None and isinstance(result, np.ndarray):
        return result.ravel()[0]
    return result

def broadcast_arrays(*args):
    args = [asarray(arg) for arg in args]
    bc_datas = np.broadcast_arrays(*args)
    bc_neparrays = []
    for i, bc_data in enumerate(bc_datas):
        valid = None
        if args[i]._valid is not None:
            valid = _broadcast_to(args[i]._valid, bc_data)
        bc_neparrays.append(_with_valid(bc_data, valid, own=False))
    return bc_neparrays

# XX This elides the casting= and preservena= arguments from Mark's
# implementation.
def copyto(dst, src, where=None):
    if where is None:
        dst[...] = src[...]
    else:
        orig_dst_shape = dst.shape
        dst, src, where = broadcast_arrays(dst, src, where)
        if orig_dst_shape != dst.shape:
            raise ValueError, "shape mismatch"
        dst[where] = src[where]

def my_assert_array_equal(x, y, *args, **kwargs):
    x = asarray(x)
    y = asarray(y)
    from numpy.testing import assert_array_equal
    assert_array_equal(x._effective_valid(), y._effective_valid(),
                       *args, **kwargs)
    assert_array_equal(_basearray(x)[x._effective_valid()],
                       _basearray(y)[y._effective_valid()],
                       *args, **kwargs)

def _monkeypatch_multiarray(exported_multiarray_functions):
    # First, replace the core C functions exported by numpy.core.multiarray
    # with our own versions:
    for funcname in exported_multiarray_functions:
        if (funcname in globals()
            and isinstance(getattr(np, funcname), type(np.array))):
            setattr(np.core.multiarray, funcname, globals()[funcname])
    # Then, delete the current references to all the modules that import from
    # numpy.core.multiarray, and re-import them. This leaves *us* with access
    # to the original functions executing in their original namespaces, while
    # everyone else will get new versions executing in new namespaces that
    # have the new multiarray functions imported.
    import sys
    modules = []
    for modname, module in sys.modules.iteritems():
        if (type(module) == type(sys)
            and (modname == "numpy" or modname.startswith("numpy."))
            and modname != "numpy.core.multiarray"):
            modules.append(modname)
    for modname in modules:
        del sys.modules[modname]
    import numpy
            
if _first_time:
    _exported_multiarray_functions = []
    for k, v in np.core.multiarray.__dict__.iteritems():
        if isinstance(v, type(np.array)) and v is getattr(np, k, None):
            _exported_multiarray_functions.append(k)
_monkeypatch_multiarray(_exported_multiarray_functions)

# Import any global constants etc. that we don't override
import numpy as patched_np
for name in patched_np.__all__:
    if name not in globals():
        globals()[name] = getattr(patched_np, name)

# This will remain False during future reload()s
_first_time = False
