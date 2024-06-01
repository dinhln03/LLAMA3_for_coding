{
	'repo_type' : 'archive',
	'download_locations' : [
		#UPDATECHECKS: http://fftw.org/download.html
		#{ "url" : "http://fftw.org/fftw-3.3.9.tar.gz", "hashes" : [ { "type" : "sha256", "sum" : "bf2c7ce40b04ae811af714deb512510cc2c17b9ab9d6ddcf49fe4487eea7af3d" }, ], },
		#{ "url" : "https://fossies.org/linux/misc/fftw-3.3.9.tar.gz", "hashes" : [ { "type" : "sha256", "sum" : "bf2c7ce40b04ae811af714deb512510cc2c17b9ab9d6ddcf49fe4487eea7af3d" }, ], },
		{ "url" : "http://fftw.org/fftw-3.3.10.tar.gz", "hashes" : [ { "type" : "sha256", "sum" : "56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467" }, ], },
		{ "url" : "https://fossies.org/linux/misc/fftw-3.3.10.tar.gz", "hashes" : [ { "type" : "sha256", "sum" : "56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467" }, ], },
	],
	'rename_folder' : 'fftw3_dll_double',
	'configure_options': '--host={target_host} --prefix={output_prefix}/fftw3_dll --exec-prefix={output_prefix}/fftw3_dll '
                         '--enable-shared --disable-static '
                         '--disable-silent-rules --disable-doc '
                         '--disable-alloca --with-our-malloc --with-windows-f77-mangling '
                         '--enable-threads --with-combined-threads '
                         '--disable-float --disable-long-double -disable-quad-precision ' # 2019.12.13 the default is "DOUBLE"
                         '--enable-sse2 --enable-avx --enable-avx2 --disable-altivec --disable-vsx --disable-neon ' # 2019.12.13 removed --enable-sse as SSE only builds with "FLOAT/SINGLE"
    ,
	'regex_replace': {
		'post_patch': [
			{
				0: r'fftw\${{PREC_SUFFIX}}\.pc',
				1: r'fftw3${{PREC_SUFFIX}}.pc',
				'in_file': './CMakeLists.txt'
			},
		],
	},
    'run_post_install' : (
		'ls -alR {output_prefix}/fftw3_dll/bin',
	),
	'update_check' : { 'url' : 'ftp://ftp.fftw.org/pub/fftw/', 'type' : 'ftpindex', 'regex' : r'fftw-(?P<version_num>[\d.]+)\.tar\.gz' },
	'_info' : { 'version' : '3.3.10', 'fancy_name' : 'fftw3_dll_double' },
}