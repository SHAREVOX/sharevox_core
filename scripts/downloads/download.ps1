#!/usr/bin/env pwsh

<#
	.DESCRIPTION
	sharevox_coreをダウンロードするためのスクリプト
#>


Param(
	[Parameter()]
	[String]
	[Alias("o")]
	# 出力先の指定
	$Output = "./sharevox_core",
	[Parameter()]
	[String]
	[Alias("v")]
	# ダウンロードするsharevox_coreのバージョンの指定
	$Version = "latest",
	[Parameter()]
	[String]
  # 追加ダウンロードするライブラリのバージョンの指定
	$AdditionalLibrariesVersion = "latest",
	[Parameter()]
	[ValidateSet("cpu","cuda","directml")]
	[string]
	# ダウンロードするデバイスを指定する(cpu,cuda,directmlを指定可能)
	$Device = "cpu",
	[Parameter()]
	[bool]
	# ダウンロードするライブラリを最小限にするように指定
	$Min = $False,
	[Parameter()]
	[ValidateSet("x86","x64")]
	[String]
	# CPUアーキテクチャの指定
	$CpuArch = ""
)
mkdir -p $Output
If (-Not(Split-Path $Output -IsAbsolute)){
  $Output=Resolve-Path $Output
}

$VoicevoxCoreRepositoryBaseUrl="https://github.com/SHAREVOX/sharevox_core"
$VoicevoxAdditionalLibrariesBaseUrl="https://github.com/VOICEVOX/voicevox_additional_libraries"
$OpenJtalkDictUrl="https://jaist.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz"
$OpenJtalkDictDirName="open_jtalk_dic_utf_8-1.11"

Function Voicevox-Core-Releases-Url($Os,$CpuArch,$Device,$Version){
	"${VoicevoxCoreRepositoryBaseUrl}/releases/download/${Version}/sharevox_core-${Os}-${CpuArch}-${Device}-${Version}.zip"
}

Function Voicevox-Additional-Libraries-Releases-Url($Os,$CpuArch,$Device,$Version){
	If ( $Device -eq "cuda" ){
		$Device="CUDA"
	} ElseIf ( $Device -eq "directml" ){
		$Device="DirectML"
	}

	"${VoicevoxAdditionalLibrariesBaseUrl}/releases/download/${Version}/${Device}-${Os}-${CpuArch}.zip"
}

Function Latest-Version($BaseUrl){
	$GetLatestUrl = "${BaseUrl}/releases/tag"
	try{Invoke-WebRequest "${BaseUrl}/releases/latest" -Method "Get" -MaximumRedirection 0 }catch{$_.Exception.Response.Headers.Location -replace "${GetLatestUrl}/","" }
}

Function Latest-Voicevox-Core-Version(){
	Latest-Version $VoicevoxCoreRepositoryBaseUrl
}

Function Latest-Voicevox-Additional-Libraries-Version(){
	Latest-Version $VoicevoxAdditionalLibrariesBaseUrl
}


Function Target-Os(){
	"windows"
}

Function Target-Arch(){
	# TODO: cpu architectureの判定を実装する
	"x64"
}

Function Download-and-Extract($Target,$Url,$ExtractDir,$ArchiveFormat){
	$TmpPath=New-TemporaryFile
	
	if ( [string]::IsNullOrEmpty($ArchiveFormat) ){
		if ( $Url.EndsWith(".tar.gz") ){
			$ArchiveFormat="tar.gz"
		} else {
			$ArchiveFormat="zip"
		}
	}

	echo "${Target}を${Url}からファイルをダウンロードします..."
	Invoke-WebRequest "${Url}" -OutFile "${TmpPath}"
	echo "${Target}をダウンロード完了,${ArchiveFormat}形式で${ExtractDir}に解凍します..."
	If( $ArchiveFormat -eq "zip" ){
		$Zip=[System.IO.Compression.ZipFile]::OpenRead("${TmpPath}")
		$Zip.Entries.ForEach{
			if ([string]::IsNullOrEmpty($_.Name)){
				return
			}
			$FixedName=$_.FullName.Replace('\', '/')
			$NewFile=[IO.FileInfo]([IO.Path]::Combine($ExtractDir,$FixedName.SubString($FixedName.IndexOf('/') + 1)))
			$NewFile.Directory.Create()
			[System.IO.Compression.ZipFileExtensions]::ExtractToFile($_,$NewFile)
		}
		$Zip.Dispose()
	}ElseIf( $ArchiveFormat -eq "tar.gz" ){
		mkdir -p "$ExtractDir"
		tar --strip-components 1 -xvzf "$TmpPath" -C "$ExtractDir"
	}
	echo "${Target}のファイルを展開完了しました"
}

$Os=Target-Os
$OpenJtalkOutput= Join-Path $Output $OpenJtalkDictDirName

If ( [string]::IsNullOrEmpty($CpuArch) ){
  $CpuArch=Target-Arch
}

If ( $Device -eq "cpu" ){
	$AdditionalLibrariesVersion=""
}

If ( $Version -eq "latest" ){
	$Version=Latest-Voicevox-Core-Version
}

If ( $AdditionalLibrariesVersion -eq "latest" ){
	$AdditionalLibrariesVersion=Latest-Voicevox-Additional-Libraries-Version
}

echo "対象OS:$Os"
echo "対象CPUアーキテクチャ:$cpu_arch"
echo "ダウンロードsharevox_coreバージョン:$version"
echo "ダウンロードアーティファクトタイプ:$Device"
echo "出力先:$Output"

$VoicevoxCoreUrl=Voicevox-Core-Releases-Url "$Os" "$CpuArch" "$Device" "$Version"
$VoicevoxAdditionalLibrariesUrl=Voicevox-Additional-Libraries-Releases-Url "$Os" "$CpuArch" "$Device" "$AdditionalLibrariesVersion"

Download-and-Extract "sharevox_core" "$VoicevoxCoreUrl" "$Output"

if ( -not $Min ){
	Download-and-Extract "open_jtalk" "$OpenJtalkDictUrl" "$OpenJtalkOutput"
	if ( -not $AdditionalLibrariesVersion -eq "" ){
		Download-and-Extract "voicevox_additional_libraries" "$VoicevoxAdditionalLibrariesUrl" "$Output"
	}
}

echo "全ての必要なファイルダウンロードが完了しました"
