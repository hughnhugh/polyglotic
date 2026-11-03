"""
Windows' audio devices as they are right now - which output and microphone
are the defaults, and every active one - straight from Windows' Core Audio
API (IMMDeviceEnumerator), through ctypes.

PortAudio (pyaudiowpatch), which the capture streams are opened with, can't
say: it reads the device list once, when it starts, so a default changed
in Windows' sound settings afterwards, or a headset switched on later,
isn't in it. overlay_core polls these to follow the defaults - and
restarts PortAudio when a device turns up that it doesn't know.

Names are the devices' friendly names - "Headphones (HyperX Cloud III
Wireless)" - the same ones PortAudio's WASAPI devices go by (its loopback
devices add " [Loopback]").
"""

import ctypes
import threading
from ctypes import POINTER, byref, c_uint, c_ushort, c_void_p, wintypes

_ole32 = ctypes.WinDLL("ole32")
_ole32.CoInitializeEx.argtypes = [c_void_p, wintypes.DWORD]
_ole32.CoInitializeEx.restype = ctypes.c_long  # checked by hand - S_FALSE (already set up) is fine
_ole32.CoCreateInstance.argtypes = [c_void_p, c_void_p, wintypes.DWORD, c_void_p, POINTER(c_void_p)]
_ole32.CoCreateInstance.restype = ctypes.c_long
_ole32.PropVariantClear.argtypes = [c_void_p]
_ole32.CLSIDFromString.argtypes = [wintypes.LPCWSTR, c_void_p]

COINIT_MULTITHREADED = 0
RPC_E_CHANGED_MODE = -2147417850   # the thread's COM is already set up, as single-threaded - usable as it is
CLSCTX_ALL = 23
E_RENDER, E_CAPTURE = 0, 1         # EDataFlow: output, input
E_MULTIMEDIA = 1                   # ERole - the default device Windows' sound settings pick (PortAudio's too)
DEVICE_STATE_ACTIVE = 1
STGM_READ = 0
VT_LPWSTR = 31


class _GUID(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16), ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text):
        guid = cls()
        _ole32.CLSIDFromString(text, byref(guid))
        return guid


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", c_ushort), ("reserved1", c_ushort), ("reserved2", c_ushort), ("reserved3", c_ushort),
                ("value", c_void_p), ("value2", c_void_p)]


_CLSID_MMDeviceEnumerator = _GUID.parse("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
_IID_IMMDeviceEnumerator = _GUID.parse("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
_PKEY_Device_FriendlyName = _PROPERTYKEY(_GUID.parse("{A45C254E-DF1C-4EFD-8020-67D146A850E0}"), 14)

_com_ready = threading.local()


def _method(obj, index, *argtypes):
    """COM method `index` of `obj` (its vtable's slot), callable with the
    rest of its arguments - raises OSError on a failed HRESULT."""
    vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)
    function = prototype(vtable[index])
    return lambda *args: function(obj, *args)


def _release(obj):
    if obj:
        vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        ctypes.WINFUNCTYPE(c_uint, c_void_p)(vtable[2])(obj)


def _enumerator():
    if not getattr(_com_ready, "done", False):
        result = _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        if result < 0 and result != RPC_E_CHANGED_MODE:
            raise OSError(f"CoInitializeEx failed: {result:#x}")
        _com_ready.done = True
    enumerator = c_void_p()
    result = _ole32.CoCreateInstance(byref(_CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
                                     byref(_IID_IMMDeviceEnumerator), byref(enumerator))
    if result < 0:
        raise OSError(f"Couldn't reach Windows' audio devices: {result & 0xFFFFFFFF:#x}")
    return enumerator


def _friendly_name(device):
    store = c_void_p()
    _method(device, 4, wintypes.DWORD, POINTER(c_void_p))(STGM_READ, byref(store))  # OpenPropertyStore
    try:
        value = _PROPVARIANT()
        _method(store, 5, POINTER(_PROPERTYKEY), POINTER(_PROPVARIANT))(  # GetValue
            byref(_PKEY_Device_FriendlyName), byref(value))
        try:
            return ctypes.wstring_at(value.value) if value.vt == VT_LPWSTR and value.value else ""
        finally:
            _ole32.PropVariantClear(byref(value))
    finally:
        _release(store)


def _default(flow):
    enumerator = _enumerator()
    try:
        device = c_void_p()
        try:
            _method(enumerator, 4, c_uint, c_uint, POINTER(c_void_p))(flow, E_MULTIMEDIA, byref(device))
        except OSError:
            return None  # none at all - nothing plugged in
        try:
            return _friendly_name(device) or None
        finally:
            _release(device)
    finally:
        _release(enumerator)


def _active(flow):
    enumerator = _enumerator()
    try:
        collection = c_void_p()
        _method(enumerator, 3, c_uint, wintypes.DWORD, POINTER(c_void_p))(  # EnumAudioEndpoints
            flow, DEVICE_STATE_ACTIVE, byref(collection))
        try:
            count = c_uint()
            _method(collection, 3, POINTER(c_uint))(byref(count))  # GetCount
            names = []
            for i in range(count.value):
                device = c_void_p()
                _method(collection, 4, c_uint, POINTER(c_void_p))(i, byref(device))  # Item
                try:
                    name = _friendly_name(device)
                finally:
                    _release(device)
                if name:
                    names.append(name)
            return sorted(names, key=str.lower)
        finally:
            _release(collection)
    finally:
        _release(enumerator)


def default_output():
    """The name of Windows' default output device - what the PC plays
    through - or None if there isn't one."""
    return _default(E_RENDER)


def default_input():
    """The name of Windows' default microphone, or None."""
    return _default(E_CAPTURE)


def outputs():
    """The names of every output device that's plugged in and enabled."""
    return _active(E_RENDER)


def inputs():
    """The names of every microphone that's plugged in and enabled."""
    return _active(E_CAPTURE)
