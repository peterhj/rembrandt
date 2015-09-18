// This file is generated. Do not edit

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]

use protobuf::Message as Message_imported_for_functions;
use protobuf::ProtobufEnum as ProtobufEnum_imported_for_functions;

#[derive(Clone,Default)]
pub struct Datum {
    // message fields
    channels: ::std::option::Option<i32>,
    height: ::std::option::Option<i32>,
    width: ::std::option::Option<i32>,
    data: ::protobuf::SingularField<::std::vec::Vec<u8>>,
    label: ::std::option::Option<i32>,
    float_data: ::std::vec::Vec<f32>,
    encoded: ::std::option::Option<bool>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::std::cell::Cell<u32>,
}

impl Datum {
    pub fn new() -> Datum {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static Datum {
        static mut instance: ::protobuf::lazy::Lazy<Datum> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const Datum,
        };
        unsafe {
            instance.get(|| {
                Datum {
                    channels: ::std::option::Option::None,
                    height: ::std::option::Option::None,
                    width: ::std::option::Option::None,
                    data: ::protobuf::SingularField::none(),
                    label: ::std::option::Option::None,
                    float_data: ::std::vec::Vec::new(),
                    encoded: ::std::option::Option::None,
                    unknown_fields: ::protobuf::UnknownFields::new(),
                    cached_size: ::std::cell::Cell::new(0),
                }
            })
        }
    }

    // optional int32 channels = 1;

    pub fn clear_channels(&mut self) {
        self.channels = ::std::option::Option::None;
    }

    pub fn has_channels(&self) -> bool {
        self.channels.is_some()
    }

    // Param is passed by value, moved
    pub fn set_channels(&mut self, v: i32) {
        self.channels = ::std::option::Option::Some(v);
    }

    pub fn get_channels<'a>(&self) -> i32 {
        self.channels.unwrap_or(0)
    }

    // optional int32 height = 2;

    pub fn clear_height(&mut self) {
        self.height = ::std::option::Option::None;
    }

    pub fn has_height(&self) -> bool {
        self.height.is_some()
    }

    // Param is passed by value, moved
    pub fn set_height(&mut self, v: i32) {
        self.height = ::std::option::Option::Some(v);
    }

    pub fn get_height<'a>(&self) -> i32 {
        self.height.unwrap_or(0)
    }

    // optional int32 width = 3;

    pub fn clear_width(&mut self) {
        self.width = ::std::option::Option::None;
    }

    pub fn has_width(&self) -> bool {
        self.width.is_some()
    }

    // Param is passed by value, moved
    pub fn set_width(&mut self, v: i32) {
        self.width = ::std::option::Option::Some(v);
    }

    pub fn get_width<'a>(&self) -> i32 {
        self.width.unwrap_or(0)
    }

    // optional bytes data = 4;

    pub fn clear_data(&mut self) {
        self.data.clear();
    }

    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }

    // Param is passed by value, moved
    pub fn set_data(&mut self, v: ::std::vec::Vec<u8>) {
        self.data = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_data<'a>(&'a mut self) -> &'a mut ::std::vec::Vec<u8> {
        if self.data.is_none() {
            self.data.set_default();
        };
        self.data.as_mut().unwrap()
    }

    // Take field
    pub fn take_data(&mut self) -> ::std::vec::Vec<u8> {
        self.data.take().unwrap_or_else(|| ::std::vec::Vec::new())
    }

    pub fn get_data<'a>(&'a self) -> &'a [u8] {
        match self.data.as_ref() {
            Some(v) => &v,
            None => &[],
        }
    }

    // optional int32 label = 5;

    pub fn clear_label(&mut self) {
        self.label = ::std::option::Option::None;
    }

    pub fn has_label(&self) -> bool {
        self.label.is_some()
    }

    // Param is passed by value, moved
    pub fn set_label(&mut self, v: i32) {
        self.label = ::std::option::Option::Some(v);
    }

    pub fn get_label<'a>(&self) -> i32 {
        self.label.unwrap_or(0)
    }

    // repeated float float_data = 6;

    pub fn clear_float_data(&mut self) {
        self.float_data.clear();
    }

    // Param is passed by value, moved
    pub fn set_float_data(&mut self, v: ::std::vec::Vec<f32>) {
        self.float_data = v;
    }

    // Mutable pointer to the field.
    pub fn mut_float_data<'a>(&'a mut self) -> &'a mut ::std::vec::Vec<f32> {
        &mut self.float_data
    }

    // Take field
    pub fn take_float_data(&mut self) -> ::std::vec::Vec<f32> {
        ::std::mem::replace(&mut self.float_data, ::std::vec::Vec::new())
    }

    pub fn get_float_data<'a>(&'a self) -> &'a [f32] {
        &self.float_data
    }

    // optional bool encoded = 7;

    pub fn clear_encoded(&mut self) {
        self.encoded = ::std::option::Option::None;
    }

    pub fn has_encoded(&self) -> bool {
        self.encoded.is_some()
    }

    // Param is passed by value, moved
    pub fn set_encoded(&mut self, v: bool) {
        self.encoded = ::std::option::Option::Some(v);
    }

    pub fn get_encoded<'a>(&self) -> bool {
        self.encoded.unwrap_or(false)
    }
}

impl ::protobuf::Message for Datum {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !try!(is.eof()) {
            let (field_number, wire_type) = try!(is.read_tag_unpack());
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::ProtobufError::WireError("unexpected wire type".to_string()));
                    };
                    let tmp = try!(is.read_int32());
                    self.channels = ::std::option::Option::Some(tmp);
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::ProtobufError::WireError("unexpected wire type".to_string()));
                    };
                    let tmp = try!(is.read_int32());
                    self.height = ::std::option::Option::Some(tmp);
                },
                3 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::ProtobufError::WireError("unexpected wire type".to_string()));
                    };
                    let tmp = try!(is.read_int32());
                    self.width = ::std::option::Option::Some(tmp);
                },
                4 => {
                    if wire_type != ::protobuf::wire_format::WireTypeLengthDelimited {
                        return ::std::result::Result::Err(::protobuf::ProtobufError::WireError("unexpected wire type".to_string()));
                    };
                    let tmp = self.data.set_default();
                    try!(is.read_bytes_into(tmp))
                },
                5 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::ProtobufError::WireError("unexpected wire type".to_string()));
                    };
                    let tmp = try!(is.read_int32());
                    self.label = ::std::option::Option::Some(tmp);
                },
                6 => {
                    try!(::protobuf::rt::read_repeated_float_into(wire_type, is, &mut self.float_data));
                },
                7 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::ProtobufError::WireError("unexpected wire type".to_string()));
                    };
                    let tmp = try!(is.read_bool());
                    self.encoded = ::std::option::Option::Some(tmp);
                },
                _ => {
                    let unknown = try!(is.read_unknown(wire_type));
                    self.mut_unknown_fields().add_value(field_number, unknown);
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        for value in self.channels.iter() {
            my_size += ::protobuf::rt::value_size(1, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        for value in self.height.iter() {
            my_size += ::protobuf::rt::value_size(2, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        for value in self.width.iter() {
            my_size += ::protobuf::rt::value_size(3, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        for value in self.data.iter() {
            my_size += ::protobuf::rt::bytes_size(4, &value);
        };
        for value in self.label.iter() {
            my_size += ::protobuf::rt::value_size(5, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        my_size += 5 * self.float_data.len() as u32;
        if self.encoded.is_some() {
            my_size += 2;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(v) = self.channels {
            try!(os.write_int32(1, v));
        };
        if let Some(v) = self.height {
            try!(os.write_int32(2, v));
        };
        if let Some(v) = self.width {
            try!(os.write_int32(3, v));
        };
        if let Some(v) = self.data.as_ref() {
            try!(os.write_bytes(4, &v));
        };
        if let Some(v) = self.label {
            try!(os.write_int32(5, v));
        };
        for v in self.float_data.iter() {
            try!(os.write_float(6, *v));
        };
        if let Some(v) = self.encoded {
            try!(os.write_bool(7, v));
        };
        try!(os.write_unknown_fields(self.get_unknown_fields()));
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields<'s>(&'s self) -> &'s ::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields<'s>(&'s mut self) -> &'s mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn type_id(&self) -> ::std::any::TypeId {
        ::std::any::TypeId::of::<Datum>()
    }

    fn as_any(&self) -> &::std::any::Any {
        self as &::std::any::Any
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for Datum {
    fn new() -> Datum {
        Datum::new()
    }

    fn descriptor_static(_: ::std::option::Option<Datum>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_singular_i32_accessor(
                    "channels",
                    Datum::has_channels,
                    Datum::get_channels,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_i32_accessor(
                    "height",
                    Datum::has_height,
                    Datum::get_height,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_i32_accessor(
                    "width",
                    Datum::has_width,
                    Datum::get_width,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_bytes_accessor(
                    "data",
                    Datum::has_data,
                    Datum::get_data,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_i32_accessor(
                    "label",
                    Datum::has_label,
                    Datum::get_label,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_f32_accessor(
                    "float_data",
                    Datum::get_float_data,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_bool_accessor(
                    "encoded",
                    Datum::has_encoded,
                    Datum::get_encoded,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<Datum>(
                    "Datum",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for Datum {
    fn clear(&mut self) {
        self.clear_channels();
        self.clear_height();
        self.clear_width();
        self.clear_data();
        self.clear_label();
        self.clear_float_data();
        self.clear_encoded();
        self.unknown_fields.clear();
    }
}

impl ::std::cmp::PartialEq for Datum {
    fn eq(&self, other: &Datum) -> bool {
        self.channels == other.channels &&
        self.height == other.height &&
        self.width == other.width &&
        self.data == other.data &&
        self.label == other.label &&
        self.float_data == other.float_data &&
        self.encoded == other.encoded &&
        self.unknown_fields == other.unknown_fields
    }
}

impl ::std::fmt::Debug for Datum {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

static file_descriptor_proto_data: &'static [u8] = &[
    0x0a, 0x11, 0x63, 0x61, 0x66, 0x66, 0x65, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x70, 0x72,
    0x6f, 0x74, 0x6f, 0x12, 0x05, 0x63, 0x61, 0x66, 0x66, 0x65, 0x22, 0x81, 0x01, 0x0a, 0x05, 0x44,
    0x61, 0x74, 0x75, 0x6d, 0x12, 0x10, 0x0a, 0x08, 0x63, 0x68, 0x61, 0x6e, 0x6e, 0x65, 0x6c, 0x73,
    0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x12, 0x0e, 0x0a, 0x06, 0x68, 0x65, 0x69, 0x67, 0x68, 0x74,
    0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x12, 0x0d, 0x0a, 0x05, 0x77, 0x69, 0x64, 0x74, 0x68, 0x18,
    0x03, 0x20, 0x01, 0x28, 0x05, 0x12, 0x0c, 0x0a, 0x04, 0x64, 0x61, 0x74, 0x61, 0x18, 0x04, 0x20,
    0x01, 0x28, 0x0c, 0x12, 0x0d, 0x0a, 0x05, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x18, 0x05, 0x20, 0x01,
    0x28, 0x05, 0x12, 0x12, 0x0a, 0x0a, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x5f, 0x64, 0x61, 0x74, 0x61,
    0x18, 0x06, 0x20, 0x03, 0x28, 0x02, 0x12, 0x16, 0x0a, 0x07, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x65,
    0x64, 0x18, 0x07, 0x20, 0x01, 0x28, 0x08, 0x3a, 0x05, 0x66, 0x61, 0x6c, 0x73, 0x65,
];

static mut file_descriptor_proto_lazy: ::protobuf::lazy::Lazy<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::lazy::Lazy {
    lock: ::protobuf::lazy::ONCE_INIT,
    ptr: 0 as *const ::protobuf::descriptor::FileDescriptorProto,
};

fn parse_descriptor_proto() -> ::protobuf::descriptor::FileDescriptorProto {
    ::protobuf::parse_from_bytes(file_descriptor_proto_data).unwrap()
}

pub fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    unsafe {
        file_descriptor_proto_lazy.get(|| {
            parse_descriptor_proto()
        })
    }
}
