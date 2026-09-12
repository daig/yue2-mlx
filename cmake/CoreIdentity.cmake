# Hash compiled engine/dependency archives, excluding frontend code and this
# generated identity object (which would otherwise create a circular digest).
set(material "")
foreach(component CORE MLX JACCL)
  file(SHA256 "${${component}}" digest)
  string(APPEND material "${component}:${digest}\n")
endforeach()
string(SHA256 identity "${material}")
file(CONFIGURE OUTPUT "${OUTPUT}" CONTENT
  "namespace lyra { const char *core_build_sha256() { return \"${identity}\"; } }\n"
  @ONLY)
file(CONFIGURE OUTPUT "${OUTPUT}.sha256" CONTENT "${identity}" @ONLY)
