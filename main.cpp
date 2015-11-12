// This code is highly based on smallpt
// http://www.kevinbeason.com/smallpt/
#include <cmath>
#include <algorithm>
#include <cassert>
#include <random>
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include "float.h"

// GLM (vector / matrix)
#define GLM_FORCE_RADIANS

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <QDebug>

const float delta = 0.1;
const float pi = 3.1415927f;
const float noIntersect = std::numeric_limits<float>::infinity();
const float lux = 2000.0;
const int maxReflexion = 2;
const float coeffombre = 1.0;
const int haa = 1;
const char* meshname= "C:\\Users/Rudi/Desktop/Gamagora/Beautiful Girl/Beautiful Girl.obj";
#define GLASS 0
#define MIRROR 1
#define DIFFUSE 2

bool isIntersect(float t)
{
    return t < noIntersect;
}

struct Ray
{
    const glm::vec3 origin, direction;
};

struct Sphere
{
    const float radius;
    const glm::vec3 center;
};


struct Triangle
{
    const glm::vec3 v0, v1, v2;
};

struct Box{
    glm::vec3 min;
    glm::vec3 max;
};

struct BoxList{

};

struct Mesh{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<int> faces;
    std::vector<int> normalIds;
    Box boundbox;

    Geometry(const glm::vec3 &center, const char* obj) {
        boundbox.min = glm::vec3(1E100, 1E100, 1E100);
        boundbox.max = glm::vec3(-1E100, -1E100, -1E100);
        FILE* f = fopen(obj, "r");
        while (!feof(f)) {
            char line[255];
            fgets(line, 255, f);
            if (line[0]=='v' && line[1]==' ') {
                glm::vec3 vec;
                sscanf(line, "v %f %f %f\n", &vec[0], &vec[2], &vec[1]);
                vec[2] = -vec[2];
                glm::vec3 p = vec*(float)50. + center;
                vertices.push_back(p);
                boundbox.max[0] = std::max(boundbox.max[0], p[0]);
                boundbox.max[1] = std::max(boundbox.max[1], p[1]);
                boundbox.max[2] = std::max(boundbox.max[2], p[2]);
                boundbox.min[0] = std::min(boundbox.min[0], p[0]);
                boundbox.min[1] = std::min(boundbox.min[1], p[1]);
                boundbox.min[2] = std::min(boundbox.min[2], p[2]);
            }
            if (line[0]=='v' && line[1]=='n') {
                glm::vec3 vec;
                sscanf(line, "vn %f %f %f\n", &vec[0], &vec[2], &vec[1]);
                vec[2] = -vec[2];
                normals.push_back(vec);
            }
            if (line[0]=='f') {
                int i0, i1, i2;
                int j0,j1,j2;
                int k0,k1,k2;
                sscanf(line, "f %u/%u/%u %u/%u/%u %u/%u/%u\n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2 );
                faces.push_back(i0-1);
                faces.push_back(i1-1);
                faces.push_back(i2-1);
                normalIds.push_back(k0-1);
                normalIds.push_back(k1-1);
                normalIds.push_back(k2-1);
            }

        }
/*
        boundingSphere.C = 0.5*(minVal+maxVal);
        boundingSphere.R = sqrt((maxVal-minVal).sqrNorm())*0.5;
*/
        fclose(f);
    }
};

    // WARRING: works only if r.d is normalized
float intersect (const Ray & ray, const Sphere &sphere)
{				// returns distance, 0 if nohit
    glm::vec3 op = sphere.center - ray.origin;		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t, b = glm:: dot(ray.direction, op), det =
        b * b - glm::dot(op, op) + sphere.radius * sphere.radius;
    if (det < 0)
        return noIntersect;
    else
        det = std::sqrt (det);
    return (t = b - det) >= 0 ? t : ((t = b + det) >= 0 ? t : noIntersect);
}

float intersect(const Ray & ray, const Triangle &triangle)
{
    auto e1 = triangle.v1 - triangle.v0;
    auto e2 = triangle.v2 - triangle.v0;

    auto h = glm::cross(ray.direction, e2);
    auto a = glm::dot(e1, h);

    auto f = 1.f / a;
    auto s = ray.origin - triangle.v0;

    auto u = f * glm::dot(s, h);
    auto q = glm::cross(s, e1);
    auto v = f * glm::dot(ray.direction, q);
    auto t = f * glm::dot(e2, q);

    if(std::abs(a) < 0.00001)
        return noIntersect;
    if(u < 0 || u > 1)
        return noIntersect;
    if(v < 0 || (u+v) > 1)
        return noIntersect;
    if(t < 0)
        return noIntersect;

    return t;
}

bool intersectBox(const Ray &r, const glm::vec3 &min, const glm::vec3 &max)
{
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    float div;

    if(r.direction.x == 0)    {
        tmin = FLT_MIN;
        tmax = FLT_MAX;
    }
    else if(r.direction.x > 0)    {
        div = 1 / r.direction.x;
        tmin = (min.x - r.origin.x) * div;
        tmax = (max.x - r.origin.x) * div;
    }
    else    {
        div = 1 / r.direction.x;
        tmin = (max.x - r.origin.x) * div;
        tmax = (min.x - r.origin.x) * div;
    }

    if(r.direction.y == 0)    {
        tymin = FLT_MIN;
        tymax = FLT_MAX;
    }
    else if(r.direction.y >= 0)    {
        div = 1 / r.direction.y;
        tymin = (min.y - r.origin.y) * div;
        tymax = (max.y - r.origin.y) * div;
    }
    else    {
        div = 1 / r.direction.y;
        tymin = (max.y - r.origin.y) * div;
        tymax = (min.y - r.origin.y) * div;
    }

    if( (tmin > tymax) || (tymin > tmax) )
        return false;

    if(tymin > tmin)
        tmin = tymin;

    if(tymax < tmax)
        tmax = tymax;

    if(r.direction.z == 0)    {
        tzmin = FLT_MIN;
        tzmax = FLT_MAX;
    }
    else if(r.direction.z > 0)    {
        div = 1 / r.direction.z;
        tzmin = (min.z - r.origin.z) * div;
        tzmax = (max.z - r.origin.z) * div;
    }
    else    {
        div = 1 / r.direction.z;
        tzmin = (max.z - r.origin.z) * div;
        tzmax = (min.z - r.origin.z) * div;
    }

    if( (tmin > tzmax) || (tzmin > tmax) )
        return false;

    if(tzmin > tmin)
        tmin = tzmin;

    if(tzmax < tmax)
        tmax = tzmax;
    return true;

}

bool intersectBox2(const Ray &r, const glm::vec3 &min, const glm::vec3 &max)
{
    double t1 = (min[0] - r.origin[0])*(-r.direction[0]);
    double t2 = (max[0] - r.origin[0])*(-r.direction[0]);

    double tmin = std::min(t1, t2);
    double tmax = std::max(t1, t2);

    for (int i = 1; i < 3; ++i) {
        t1 = (min[i] - r.origin[i])*(-r.direction[i]);
        t2 = (max[i] - r.origin[i])*(-r.direction[i]);

        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));
    }

    return tmax > std::max(tmin, 0.0);
}

float intersect (const Ray & ray, const Mesh &mesh){
    if(!intersectBox(ray, mesh.boundbox.min, mesh.boundbox.max))return noIntersect;
    float pluproche = noIntersect;
    for(int i=0; i<mesh.faces.size()-2; i++){
        Triangle titi{mesh.vertices.at(mesh.faces.at(i)),mesh.vertices.at(mesh.faces.at(i+1)),mesh.vertices.at(mesh.faces.at(i+2))};
        float tutu = intersect(ray, titi);
        if(tutu!=noIntersect){
            if(tutu<pluproche)pluproche=tutu;
        }
    }
    return pluproche;
}

struct Diffuse
{
    const glm::vec3 color;
};

struct Glass
{
    const glm::vec3 color;
};

struct Mirror
{
    const glm::vec3 color;
};

template<typename T>
glm::vec3 albedo(const T &t)
{
    return t.color;
}

glm::vec3 getnormal(const glm::vec3 &point, const Mesh &mesh){
    return glm::vec3(0.0,1.0,1);
}


glm::vec3 getnormal(const glm::vec3 &point, const Sphere &sphere){
    return glm::normalize(point - sphere.center);
}

glm::vec3 getnormal(const glm::vec3 &point, const Triangle &triangle){
    return glm::normalize(glm::cross((triangle.v1-triangle.v0),(triangle.v2-triangle.v0)));
}

float reflection(const glm::vec3 &c, const glm::vec3 &n, const glm::vec3 &l, const Diffuse){
    return glm::normalize(glm::dot(n,l)/pi);
}

float reflection(const glm::vec3 &c, const glm::vec3 &n, const glm::vec3 &l, const Glass){
    return 1;
}

float reflection(const glm::vec3 &c, const glm::vec3 &n, const glm::vec3 &l, const Mirror){
    return 1;
}

int materiel(const Mirror){
    return MIRROR;
}
int materiel(const Diffuse){
    return DIFFUSE;
}
int materiel(const Glass){
    return GLASS;
}

glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l, const Diffuse mat);
glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l, const Glass mat);
glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l, const Mirror mat);
glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations, const Diffuse mat);
glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations, const Glass mat);
glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations, const Mirror mat);
glm::vec3 radiance (const Ray & r, int nbiterations);

struct Object
{
    virtual float intersect(const Ray &r) const = 0;
    virtual glm::vec3 albedo() const = 0;
    virtual glm::vec3 getnormal(const glm::vec3 &point) const = 0;
    virtual float reflection(const glm::vec3 &c, const glm::vec3 &n, const glm::vec3 &l)const =0;
    virtual glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l)const =0;
    virtual glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations)const =0;
    virtual int materiel()const =0;
};

template<typename P, typename M>
struct ObjectTpl final : Object
{
    ObjectTpl(const P &_p, const M &_m)
        :primitive(_p), material(_m)
    {}

    float intersect(const Ray &ray) const
    {
        return ::intersect(ray, primitive);
    }

    glm::vec3 albedo() const
    {
        return ::albedo(material);
    }

    glm::vec3 getnormal(const glm::vec3 &point) const
    {
        return ::getnormal(point, primitive);
    }

    glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l) const
    {
        return ::direct(impactpoint, c, n, l, material);
    }

    glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations)const
    {
        return ::indirect(r, normale, impactpoint, nbiterations, material);
    }

    int materiel()const{
        return ::materiel(material);
    }

    float reflection(const glm::vec3 &c, const glm::vec3 &n, const glm::vec3 &l)const{
        return ::reflection(c, n, l,material);
    }


    const P &primitive;
    const M &material;
};


template<typename P, typename M>
std::unique_ptr<Object> makeObject(const P&p, const M&m)
{
    return std::unique_ptr<Object>(new ObjectTpl<P, M>{p, m});
}

// Scene
namespace scene
{
    // Primitives

    // Left Wall
    const Triangle leftWallA{{0, 0, 0}, {0, 100, 0}, {0, 0, 150}};
    const Triangle leftWallB{{0, 100, 150}, {0, 0, 150}, {0, 100, 0}};

    // Right Wall
    const Triangle rightWallA{{100, 0, 0}, {100, 0, 150}, {100, 100, 0}};
    const Triangle rightWallB{{100, 100, 150}, {100, 100, 0}, {100, 0, 150}};

    // Back wall
    const Triangle backWallA{{0, 0, 0}, {100, 0, 0}, {100, 100, 0}};
    const Triangle backWallB{{0, 0, 0}, {100, 100, 0}, {0, 100, 0}};

    // Bottom Floor
    const Triangle bottomWallA{{0, 0, 0}, {100, 0, 150}, {100, 0, 0}};
    const Triangle bottomWallB{{0, 0, 0}, {0, 0, 150}, {100, 0, 150}};

    // Top Ceiling
    const Triangle topWallA{{0, 100, 0}, {100, 100, 0}, {0, 100, 150}};
    const Triangle topWallB{{100, 100, 150}, {0, 100, 150}, {100, 100, 0}};

    const Sphere topSphere{8.0, glm::vec3 {80, 60, 60}};
    const Sphere leftSphere{16.5, glm::vec3 {27, 16.5, 47}};
    const Sphere rightSphere{16.5, glm::vec3 {73, 16.5, 78}};
    const Sphere midSphere{12.5, glm::vec3 {27, 50, 100}};

    const glm::vec3 light{50, 66, 81.6};

    // Mesh
    Mesh mesh2;

    // Materials
    const Diffuse white{{.75, .75, .75}};
    const Diffuse red{{.75, .25, .25}};
    const Diffuse blue{{.25, .25, .75}};
    const Diffuse yellow{{.75, .75, .0}};

    const Glass glass{{.99, .99, .99}};
    const Mirror mirror{{.99, .99, .99}};

    // Objects
    // Note: this is a rather convoluted way of initialising a vector of unique_ptr ;)
    const std::vector<std::unique_ptr<Object>> objects = [] (){
        std::vector<std::unique_ptr<Object>> ret;
        ret.push_back(makeObject(backWallA, white));
        ret.push_back(makeObject(backWallB, white));
        ret.push_back(makeObject(topWallA, white));
        ret.push_back(makeObject(topWallB, white));
        ret.push_back(makeObject(bottomWallA, white));
        ret.push_back(makeObject(bottomWallB, white));
        ret.push_back(makeObject(rightWallA, blue));
        ret.push_back(makeObject(rightWallB, blue));
        ret.push_back(makeObject(leftWallA, red));
        ret.push_back(makeObject(leftWallB, red));

        ret.push_back(makeObject(mesh2, yellow));

        //ret.push_back(makeObject(topSphere, mirror));
        //ret.push_back(makeObject(leftSphere, mirror));
        //ret.push_back(makeObject(rightSphere, glass));
        //ret.push_back(makeObject(midSphere, glass));

        return ret;
    }();
}

thread_local std::default_random_engine generator;
thread_local std::uniform_real_distribution<float> distribution(0.0,1.0);

float random_u()
{
    return distribution(generator);
}

glm::vec3 sample_cos(const float u, const float v, const glm::vec3 n)
{
    // Ugly: create an ornthogonal base
    glm::vec3 basex, basey, basez;

    basez = n;
    basey = glm::vec3(n.y, n.z, n.x);

    basex = glm::cross(basez, basey);
    basex = glm::normalize(basex);

    basey = glm::cross(basez, basex);

    // cosinus sampling. Pdf = cosinus
    return  basex * (std::cos(2.f * pi * u) * std::sqrt(1.f - v)) +
        basey * (std::sin(2.f * pi * u) * std::sqrt(1.f - v)) +
        basez * std::sqrt(v);
}

int toInt (const float x)
{
    return int (std::pow (glm::clamp (x, 0.f, 1.f), 1.f / 2.2f) * 255 + .5);
}

// WARNING: ASSUME NORMALIZED RAY
// Compute the intersection ray / scene.
// Returns true if intersection
// t is defined as the abscisce along the ray (i.e
//             p = r.o + t * r.d
// id is the id of the intersected object
Object* intersect (const Ray & r, float &t)
{
    t = noIntersect;
    Object *ret = nullptr;

    for(auto &object : scene::objects)
    {
        float d = object->intersect(r);
        if (isIntersect(d) && d < t)
        {
            t = d;
            ret = object.get();
        }
    }

    return ret;
}

// Reflect the ray i along the normal.
// i should be oriented as "leaving the surface"
glm::vec3 reflect(const glm::vec3 i, const glm::vec3 n)
{
    return n * (glm::dot(n, i)) * 2.f - i;
}

float sin2cos (const float x)
{
    return std::sqrt(std::max(0.0f, 1.0f-x*x));
}

// Fresnel coeficient of transmission.
// Normal point outside the surface
// ior is n0 / n1 where n0 is inside and n1 is outside
float fresnelR(const glm::vec3 i, const glm::vec3 n, const float ior)
{
    if(glm::dot(n, i) < 0)
        return fresnelR(i, n * -1.f, 1.f / ior);

    float R0 = (ior - 1.f) / (ior + 1.f);
    R0 *= R0;

    return R0 + (1.f - R0) * std::pow(1.f - glm::dot(i, n), 5.f);
}

// compute refraction vector.
// return true if refraction is possible.
// i and n are normalized
// output wo, the refracted vector (normalized)
// n point oitside the surface.
// ior is n00 / n1 where n0 is inside and n1 is outside
//
// i point outside of the surface
bool refract(glm::vec3 i, glm::vec3 n, float ior, glm::vec3 &wo)
{
    i = i * -1.f;

    if(glm::dot(n, i) > 0)
    {
        n = n * -1.f;
    }
    else
    {
        ior = 1.f / ior;
    }

    float k = 1.f - ior * ior * (1.f - glm::dot(n, i) * glm::dot(n, i));
    if (k < 0.)
        return false;

    wo = i * ior - n * (ior * glm::dot(n, i) + std::sqrt(k));

    return true;
}

glm::vec3 sample_sphere(const float r, const float u, const float v, float &pdf, const glm::vec3 normal)
{
    pdf = 1.f / (pi * r * r);
    glm::vec3 sample_p = sample_cos(u, v, normal);

    float cos = glm::dot(sample_p, normal);

    pdf *= cos;
    return sample_p * r;
}

glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l, const Diffuse mat){
    glm::vec3 color;
    float disttolight = glm::distance(impactpoint,scene::light);
    float coefflight = glm::dot(l,n);
    coefflight *= lux;
    coefflight /= (disttolight*disttolight);
    color = mat.color * glm::abs(coefflight);

    //2eme rayon ========= ombres ==================

    glm::vec3 color2 = glm::vec3(0,0,0);
    float pdf;
    glm::vec3 lightpoint = scene::light + sample_sphere(4,random_u(),random_u(),pdf,glm::normalize(l));
    glm::vec3 dirtolight = glm::normalize(lightpoint - impactpoint);
    disttolight = glm::distance(impactpoint,scene::light);
    Ray newray{impactpoint+(delta*dirtolight), dirtolight};
    float distinter;
    intersect(newray,distinter);
    if(distinter<disttolight);
    else{
        color2 = mat.color * glm::abs(coefflight);
    }
    color = color2;
    return color;
}

glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l, const Glass mat){
    glm::vec3 color;
    float disttolight = glm::distance(impactpoint,scene::light);
    float coefflight = glm::dot(l,n);
    coefflight *= lux/10;
    coefflight /= (disttolight*disttolight);
    color = mat.color * glm::abs(coefflight);

    //2eme rayon ========= ombres ==================

    glm::vec3 color2 = glm::vec3(0,0,0);
    float pdf;
    glm::vec3 lightpoint = scene::light + sample_sphere(4,random_u(),random_u(),pdf,glm::normalize(l));
    glm::vec3 dirtolight = glm::normalize(lightpoint - impactpoint);
    disttolight = glm::distance(impactpoint,scene::light);
    Ray newray{impactpoint+(delta*dirtolight), dirtolight};
    float distinter;
    intersect(newray,distinter);
    if(distinter<disttolight);
    else{
        color2 = mat.color * glm::abs(coefflight);
    }
    color = color2;
    return color;
}

glm::vec3 direct(const glm::vec3 impactpoint, const glm::vec3 c, const glm::vec3 n, const glm::vec3 l, const Mirror mat){
    glm::vec3 color;
    float disttolight = glm::distance(impactpoint,scene::light);
    float coefflight = glm::dot(l,n);
    coefflight *= lux/10;
    coefflight /= (disttolight*disttolight);
    color = mat.color * glm::abs(coefflight);

    //2eme rayon ========= ombres ==================

    glm::vec3 color2 = glm::vec3(0,0,0);
    float pdf;
    glm::vec3 lightpoint = scene::light + sample_sphere(4,random_u(),random_u(),pdf,glm::normalize(l));
    glm::vec3 dirtolight = glm::normalize(lightpoint - impactpoint);
    disttolight = glm::distance(impactpoint,scene::light);
    Ray newray{impactpoint+(delta*dirtolight), dirtolight};
    float distinter;
    intersect(newray,distinter);
    if(distinter<disttolight);
    else{
        color2 = mat.color * glm::abs(coefflight);
    }
    color = color2;
    return color;
}

glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations, const Diffuse mat){
    //return glm::vec3(0,0,0);
    if(nbiterations ==0)return glm::vec3(0,0,0);
    glm::vec3 w = sample_cos(random_u(),random_u(),normale);
    glm::vec3 newdirectionReflexion = w;
    Ray newrayreflexion{impactpoint+(newdirectionReflexion*delta), newdirectionReflexion};
    return (radiance(newrayreflexion, nbiterations-1)*mat.color);
}

glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations, const Glass mat){
    if(nbiterations ==0)return glm::vec3(0,0,0);
    glm::vec3 newdirectionRefract;
    if(refract(-r.direction,normale,1.33,newdirectionRefract)){
        glm::vec3 newdirectionReflexion = reflect(-r.direction,normale);
        Ray newrayrefraxion{impactpoint+(newdirectionRefract*delta), newdirectionRefract};
        Ray newrayreflexion{impactpoint+(newdirectionReflexion*delta), newdirectionReflexion};
        float coeffresnel = fresnelR(-r.direction, normale, 1.33);
        return (coeffresnel*mat.color*radiance(newrayreflexion, nbiterations-1))+
               ((1.0f-coeffresnel)*mat.color*radiance(newrayrefraxion, nbiterations-1));

    }
    return glm::vec3(0,0,0);
}

glm::vec3 indirect(const Ray & r, const glm::vec3 normale, const glm::vec3 impactpoint, int nbiterations, const Mirror mat){
    if(nbiterations ==0)return glm::vec3(0,0,0);
    glm::vec3 newdirection = reflect(-r.direction,normale);
    Ray newrayreflexion{impactpoint+(newdirection*delta), newdirection};
    return mat.color*radiance(newrayreflexion, nbiterations-1);
}

glm::vec3 radiance (const Ray & r, int nbiterations)
{
    float distinter;
    Object *obj = intersect(r, distinter);
    if(obj == nullptr){
        return glm::vec3(0,0,0);
    }
    glm::vec3 impactpoint = (r.origin + r.direction*distinter);
    return obj->direct(impactpoint, -r.direction,obj->getnormal(impactpoint),glm::normalize(scene::light-impactpoint))+
           obj->indirect(r,obj->getnormal(impactpoint),impactpoint,nbiterations);
}

float integrale(){
    float result =0;
    for(int i=0; i<100;i++){
        result += cos((pi*random_u())-pi/2)*pi;
    }
    return result /= 100;
}

double integrale2(){
    double result =0;
    for(int i=0; i<100;i++){
        double u = random_u();
        double v = random_u();
        double x = sqrt(-2*log(u));
        x *= cos(2*pi*v)*0.7;
        if(x < -(pi/2));
        else if(x > (pi/2));
        else{
            double sigmacarre = 0.7*0.7;
            double expo = exp((-(x*x))/(2*sigmacarre));
            double pdf = expo / (0.7*sqrt(2*pi));
            result += (cos(x)*cos(x))/pdf;
        }
    }
    return result /= 100;
}

int main (int, char **)
{
    scene::mesh2.Geometry(glm::vec3 {75, 0.0, 25},meshname);
    /*
    qDebug()<<integrale();
    qDebug()<<integrale2();*/
    int w = 768, h = 768;
    std::vector<glm::vec3> colors(w * h, glm::vec3{0.f, 0.f, 0.f});

    Ray cam {{50, 52, 295.6}, glm::normalize(glm::vec3{0, -0.042612, -1})};	// cam pos, dir
    float near = 1.f;
    float far = 10000.f;

    glm::mat4 camera =
        glm::scale(glm::mat4(1.f), glm::vec3(float(w), float(h), 1.f))
        * glm::translate(glm::mat4(1.f), glm::vec3(0.5, 0.5, 0.f))
        * glm::perspective(float(54.5f * pi / 180.f), float(w) / float(h), near, far)
        * glm::lookAt(cam.origin, cam.origin + cam.direction, glm::vec3(0, 1, 0))
        ;

    glm::mat4 screenToRay = glm::inverse(camera);


    for (int y = 0; y < h; y++)
    {
        std::cerr << "\rRendering: " << 100 * y / (h - 1) << "%";
        #pragma omp parallel for schedule(dynamic,1)
        for (unsigned short x = 0; x < w; x++)
        {
            glm::vec3 r =glm::vec3(0,0,0);
            for(int aa =0; aa<haa; aa++){
                float u = random_u();
                float v = random_u();
                float r2 = sqrt(-2*log(u));

                float x2 =r2*cos(2*pi*v)*0.5;
                float y2 =r2*sin(2*pi*v)*0.5;

                glm::vec4 p0 = screenToRay * glm::vec4{float(x), float(h - y), 0.f, 1.f};
                glm::vec4 p1 = screenToRay * glm::vec4{float(x+x2), float(h - (y+y2)), 1.f, 1.f};

                glm::vec3 pp0 = glm::vec3(p0 / p0.w);
                glm::vec3 pp1 = glm::vec3(p1 / p1.w);

                glm::vec3 d = glm::normalize(pp1 - pp0);
                r += radiance (Ray{pp0, d},maxReflexion);
            }
            r/=(float)haa;
            colors[y * w + x] += glm::clamp(r, glm::vec3(0.f, 0.f, 0.f), glm::vec3(1.f, 1.f, 1.f));
        }
    }

    {
        std::fstream f("image.ppm", std::fstream::out);
        f << "P3\n" << w << " " << h << std::endl << "255" << std::endl;

        for (auto c : colors)
            f << toInt(c.x) << " " << toInt(c.y) << " " << toInt(c.z) << " ";
    }
}

